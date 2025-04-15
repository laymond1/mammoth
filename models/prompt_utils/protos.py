from collections import defaultdict
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from sklearn.cluster import SpectralClustering, KMeans


class ProtoDataset(Dataset):
    def __init__(self, args, prototypes, prototypes_var, classes):
        self.args = args
        self.prototypes = prototypes
        self.prototypes_var = prototypes_var
        self.classes = classes
        assert len(self.prototypes) == len(self.prototypes_var) == len(classes)

        # if self.args.use_mc_proto:
        #     assert type(self.prototypes) == list
        #     assert type(self.prototypes_var) == list
        #     index_mapping = []
        #     for idx in range(len(classes)):
        #         cur_protos = self.prototypes[idx].squeeze()
        #         cur_mapping = torch.full((len(cur_protos), 1), idx)
        #         index_mapping.append(cur_mapping)
        #     self.prototypes = torch.cat(self.prototypes, dim=0).cuda()
        #     self.prototypes_var = torch.cat(self.prototypes_var, dim=0).cuda()
        #     self.index_mapping = torch.cat(index_mapping, dim=0).squeeze().cuda()
        # else:
        assert type(self.prototypes) == list
        assert type(self.prototypes_var) == list
        self.prototypes = torch.stack(self.prototypes, dim=0).cuda()
        self.prototypes_var = torch.stack(self.prototypes_var, dim=0).cuda()

        self.scale = torch.sqrt(self.prototypes_var)  # (proto_num, embeding_dim)
        assert len(self.scale) == len(self.prototypes)

    def __len__(self):
        return self.prototypes.size(0)

    def __getitem__(self, idx):
        if self.args.proto_trans:
            proto_aug = self.proto_transform(idx)
        else:
            proto_aug = self.prototypes[idx]
        # if self.args.use_mc_proto:
        #     label = self.classes[self.index_mapping[idx]]
        # else:
        label = self.classes[idx]
        return proto_aug, label

    def proto_transform(self, idx):  
        proto = self.prototypes[idx]
        gaussian_noise = torch.normal(torch.zeros(len(proto)), 1).cuda() * self.scale[idx]
        proto_aug = proto + gaussian_noise
        return proto_aug
    
    def get_item_by_labels(self, labels):
        """
        Get items by labels.
        labels: (N,) torch.Tensor (class labels)
        return: (N, feature_dim) torch.Tensor
        """
        protos = torch.empty((0, self.prototypes.shape[1]), device=self.prototypes.device)
        # 8, 1, 0, 2, 4
        for cls in labels:
            protos = torch.cat([protos, self.__getitem__(cls)[0].unsqueeze(0)], dim=0)

        return protos, labels

  
class Prototypes:
    def __init__(self, args, feature_dim, device):
        """
        Prototypes with Cumulative Average using PyTorch.
        feature_dim: Feature vector dimension
        device: Device to run the classifier on
        Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        self.args = args
        self.device = device
        self.feature_dim = feature_dim

        self.class_sums = defaultdict(lambda: torch.zeros(feature_dim, device=self.device))  # Class feature sum
        self.class_var_sums = defaultdict(lambda: torch.zeros(feature_dim, device=self.device))  # Class feature variance sum
        self.class_counts = defaultdict(lambda: torch.tensor(0, dtype=torch.float32, device=self.device))  # Class sample count
        self.class_global_prototypes = {}  # Global prototype mean
        self.class_global_prototypes_var = {}  # Global prototype variance
        self.class_prototypes = {}  # Class prototype mean
        self.class_mc_prototypes = {}  # Class multi-centroid prototype mean
        self.class_prototypes_var = {}  # Class prototype variance
        self.class_mc_prototypes_var = {}  # Class multi-centroid prototype variance

    def update(self, features, labels):
        """
        Update class prototypes with new data.
        features: (N, feature_dim) torch.Tensor (input feature vectors)
        labels: (N,) torch.Tensor (class labels)
        """
        with torch.no_grad():
            for x, y in zip(features, labels):
                y = y.item()  # Convert label tensor to int
                self.class_sums[y] += x  # Update sum of feature vectors for class
                self.class_var_sums[y] += x ** 2  # Update sum of squared feature vectors for class
                self.class_counts[y] += 1  # Increment count of class instances
                
                # # Compute cumulative mean
                # mean = self.class_sums[y] / self.class_counts[y]
                # self.class_prototypes[y] = mean

                # # Compute cumulative variance using Welford's algorithm
                # if self.class_counts[y] > 1:
                #     delta = x - self.class_prototypes[y]  # Difference from mean
                #     self.class_var_sums[y] += delta * (x - mean)  # Welford's update
                #     var = self.class_var_sums[y] / (self.class_counts[y] - 1)  # Unbiased variance
                #     var = torch.clamp(var, min=0)  # Prevent negative variance
                # else:
                #     var = torch.zeros_like(mean)  # Variance is zero for a single sample
                # self.class_prototypes_var[y] = var

            # Compute multi-centroid prototypes
            # if self.args.use_mc_proto:
            #     mc_proto, mc_proto_var, mc_proto_sim, mc_features = gen_single_mc_proto(self.args, features)
            #     self.class_mc_prototypes[y] = mc_proto
            #     self.class_mc_prototypes_var[y] = mc_proto_var

    def get_global_prototypes(self):
        """
        Return global prototypes.
        """
        for y in self.class_prototypes.keys():
            self.class_global_prototypes[y] = self.class_sums[y] / self.class_counts[y]
            self.class_global_prototypes_var[y] = self.class_var_sums[y] / (self.class_counts[y] - 1)
        return self.class_global_prototypes, self.class_global_prototypes_var

    def predict(self, features):
        """
        Predict class for input feature vectors (optimized version).
        features: (N, feature_dim) torch.Tensor
        return: (N,) torch.Tensor (predicted classes)
        """
        # features = features.to(self.device)
        
        # Stack prototypes into a matrix (num_classes, feature_dim)
        class_labels = list(self.class_prototypes.keys())
        class_prototypes = torch.stack([self.class_prototypes[cls] for cls in class_labels]).to(self.device)  # (num_classes, feature_dim)

        if self.args.similarity == 'cosine':
            # Normalize features and prototypes
            features_norm = torch.nn.functional.normalize(features, dim=1)
            prototypes_norm = torch.nn.functional.normalize(class_prototypes, dim=1)
            # Compute cosine similarity (batch matrix multiplication)
            similarity_matrix = torch.mm(features_norm, prototypes_norm.T)  # (N, num_classes)
            pred_indices = similarity_matrix.argmax(dim=1)  # Higher cosine similarity = more similar

        elif self.args.similarity == 'l2':
            # Compute squared L2 distance (using broadcasting)
            similarity_matrix = torch.cdist(features, class_prototypes, p=2)  # (N, num_classes)
            pred_indices = similarity_matrix.argmin(dim=1)  # Lower L2 distance = more similar

        elif self.args.similarity == 'l1':
            # Compute L1 (Manhattan) distance (using broadcasting)
            similarity_matrix = torch.cdist(features, class_prototypes, p=1)  # (N, num_classes)
            pred_indices = similarity_matrix.argmin(dim=1)  # Lower L1 distance = more similar

        else:
            raise ValueError(f"Invalid similarity metric: {self.args.similarity}")

        # Convert indices to class labels
        predictions = torch.tensor([class_labels[idx] for idx in pred_indices], device=self.device)

        return predictions

    def get_prototypes(self, exposed_classes):
        """
        Return class prototypes.
        """
        return torch.stack([self.class_prototypes[y] for y in exposed_classes], dim=0)
        # return {cls: proto.clone().detach().cpu() for cls, proto in self.class_prototypes.items()}
    
    def generate_proto_data(self, labels, n_seen_classes):
        # convert to list
        prototypes = []
        prototypes_var = []
        for cls in range(n_seen_classes):
            prototypes.append(self.class_global_prototypes[cls])
            prototypes_var.append(self.class_global_prototypes_var[cls])
        proto_dataset = ProtoDataset(self.args, prototypes, prototypes_var, list(range(n_seen_classes)))
        return proto_dataset.get_item_by_labels(labels)

    def generate_proto_dataloader(self, classes_up2now):
        """
        Generate prototypes for contrastive prototype loss.
        """
        # assert len(self.class_prototypes.keys()) == len(self.class_prototypes_var.keys()) == len(classes_up2now)

        # convert to list
        prototypes = []
        prototypes_var = []
        for cls in classes_up2now:
            prototypes.append(self.class_prototypes[cls])
            prototypes_var.append(self.class_prototypes_var[cls])
        if self.args.use_mc_proto:
            proto_dataset = ProtoDataset(self.args, prototypes, prototypes_var, classes_up2now)
        else:
            proto_dataset = ProtoDataset(self.args, prototypes, prototypes_var, classes_up2now)

        proto_sampler = RandomSampler(proto_dataset, replacement=True, num_samples=self.args.proto_aug_bs)
        proto_dataloader = DataLoader(dataset=proto_dataset,
                                    num_workers=0,
                                    batch_size=self.args.proto_aug_bs,
                                    sampler=proto_sampler,
                                    shuffle=False,
                                    drop_last=False)
        return proto_dataloader


# ETC
def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Safe calculation of matrix multiplication.
    If input is float16, will cast to float32 for computation and back again.
    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return torch.matmul(x.float(), y.float().t()).half()
    return torch.matmul(x, y.t())


def cosine_similarity(input, target):
    """
    input: (dim_input, embed_dim)
    target: (dim_ouput, embed_dim)
    similarity: (dim_input, dim_ouput)
    """
    input_norm = nn.functional.normalize(input, dim=1, p=2)
    target_norm = nn.functional.normalize(target, dim=1, p=2)
    similarity = _safe_matmul(input_norm, target_norm)
    similarity  = torch.nan_to_num(similarity)
    eps = 1e-8
    similarity[similarity<=eps] = eps
    return similarity


def gen_single_mc_proto(args, input_data, clustering):
    cur_proto, cur_proto_var, cur_proto_sim, cur_features = [], [], [], []
    if args.gen_proto_mode == 'spectral':
        affinity_matrix = cosine_similarity(input_data, input_data)
        affinity_matrix = affinity_matrix.cpu().detach().numpy()
        clustering.fit_predict(affinity_matrix)
    elif args.gen_proto_mode == 'kmeans':
        # clustering.fit(input_data.cpu().numpy())
        clustering.partial_fit(input_data.cpu().numpy())
    else:
        raise NotImplementedError

    for label in range(args.mc_num):
        feature = input_data[clustering.labels_ == label, :]
        if not torch.is_tensor(feature):
            feature = torch.tensor(feature).cuda()
        var, mean = torch.var_mean(feature, dim=0)
        sim = cosine_similarity(mean.unsqueeze(0), feature)
        cur_proto.append(mean)
        cur_proto_var.append(var)
        cur_proto_sim.append(sim)
        cur_features.append(feature)
    cur_proto = torch.stack(cur_proto, dim=0)
    cur_proto_var = torch.stack(cur_proto_var, dim=0)
    return cur_proto, cur_proto_var, cur_proto_sim, cur_features