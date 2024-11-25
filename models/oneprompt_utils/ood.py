# This code is part of the OVOR model.

import math

import faiss
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset


class NPOS:
    def __init__(self, args):
        self.knn_idx = faiss.IndexFlatL2(768)
        self.K = getattr(args, 'K', 50)
        self.sample_from = getattr(args, 'sample_from', 600)
        self.select = getattr(args, 'select', 50) 
        self.cov_mat = getattr(args, 'cov', 1.0)
        self.per_class = getattr(args, 'num_per_class', 40)
        self.dis = MultivariateNormal(torch.zeros(768), torch.eye(768))
        self.id_bsz = getattr(args, 'id_bsz', 64)
        self.thres_id = getattr(args, 'thres_id', -15.0)
        self.thres_ood = getattr(args, 'thres_ood', -3.0)
        self.lmda = getattr(args, 'lmda', 0.1)
        self.huber = torch.nn.HuberLoss()#.cuda() if config.get('huber', True) else torch.nn.MSELoss()#.cuda()
    
    def generate(self, in_dist, targets):
        num_cls = targets.unique().size(0)
        id_count = in_dist.size(0)
        oods = self._generate(in_dist, num_cls)
        id_loader = DataLoader(dataset=TensorDataset(in_dist, targets), batch_size=self.id_bsz, shuffle=True)
        num_batch = int(math.ceil(id_count / self.id_bsz))
        ood_bsz = int(math.floor(oods.shape[0] / num_batch))
        ood_loader = DataLoader(dataset=TensorDataset(oods), batch_size=ood_bsz)
        return id_loader, ood_loader

    def _generate(self, in_dist, num_cls):
        offsets = self.dis.rsample((self.sample_from,))
        normed = in_dist / torch.norm(in_dist, p=2, dim=1, keepdim=True)
        select = min(self.select, normed.shape[0])
        rand_ind = np.random.choice(normed.shape[0], replace=False)
        self.knn_idx.add(normed.numpy())
        boundary_ids = self._boundary(normed, select)
        boundary_points = torch.cat([in_dist[i:i+1].repeat(self.sample_from, 1) for i in boundary_ids])
        ood_list = self.cov_mat * offsets.repeat(select, 1)
        oods = self._select_ood(ood_list, num_cls)
        self.knn_idx.reset()
        return oods

    def _boundary(self, ids, select):
        distance, _ = self.knn_idx.search(ids.numpy(), self.K)
        _, min_ids = torch.topk(torch.tensor(distance[:, -1]), select)
        return min_ids

    def _select_ood(self, ood_list, num_cls):
        distance, _ = self.knn_idx.search(ood_list.numpy(), self.K)
        k_th_distance = torch.tensor(distance[:, -1])
        num_points = min(k_th_distance.shape[0], self.per_class * num_cls)
        _, min_ids = torch.topk(k_th_distance, num_points, dim=0)
        min_ids = min_ids.squeeze()
        return ood_list[min_ids]

    def loss(self, id_logits, ood_logits):
        id_score = -torch.logsumexp(id_logits, dim=1)
        ood_score = -torch.logsumexp(ood_logits, dim=1)
        id_dist = F.relu(id_score - self.thres_id)
        ood_dist = F.relu(self.thres_ood - ood_score)
        return (
            self.lmda * (
                self.huber(id_dist, torch.zeros_like(id_dist)) +
                self.huber(ood_dist, torch.zeros_like(ood_dist))
            ),
            id_score.mean(),
            ood_score.mean()
        )