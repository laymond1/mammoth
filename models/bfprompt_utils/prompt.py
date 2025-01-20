import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BFPrompt(nn.Module):
    def __init__(self, args, emb_d, cls_per_prompt, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.cls_per_prompt = cls_per_prompt
        self._init_smart(emb_d)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}',p)
        # single key
        k = tensor_prompt(self.e_pool_size, self.key_d)
        setattr(self, f'e_k',k)

    def _init_smart(self, emb_d):
        
        self.top_k = self.args.top_k
        self.cls_to_prompt = True

        # prompt locations
        self.g_layers = self.args.g_prompt_layer_idx
        self.e_layers = self.args.e_prompt_layer_idx

        # prompt pool size
        self.g_p_length = self.args.g_prompt_length
        self.e_p_length = self.args.e_prompt_length
        self.e_pool_size = self.args.e_prompt_pool_size
        self.temperature = self.args.temperature
        self.loss = SupConLoss(temperature=self.temperature)

        # init
        self.register_buffer('train_count', torch.zeros(self.e_pool_size))
        self.register_buffer('eval_count', torch.zeros(self.e_pool_size))
        self.register_buffer('gt_count', torch.zeros(self.e_pool_size))
        self.top_k_idx = None

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, label=None, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k') # single key
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.cls_to_prompt:
                    # select prompt to train by label
                    prompt_indices = label2prompt(label, cls_per_prompt=self.cls_per_prompt) # -> Torch.Tensor
                    # count selected prompt when trianing
                    with torch.no_grad():
                        num = prompt_indices.view(-1).bincount(minlength=self.e_pool_size)
                        self.train_count += num
                    P_ = p[prompt_indices.view(-1, 1)]
                    
                    if self.args.use_supcon:
                        # loss = supervised_contrastive_loss(cos_sim, prompt_indices, temperature=self.temperature)
                        # loss = self.loss(cos_sim, prompt_indices)
                        distance = cos_sim
                        key_wise_distance = 1 - F.cosine_similarity(n_K.unsqueeze(1), n_K.detach(), dim=-1)
                        key_dist_topk = key_wise_distance[prompt_indices].clone()
                        loss = - ((key_dist_topk.exp().sum() / (distance.exp().sum() + key_dist_topk.exp().sum()) + 1e-6).log())
                    else:
                        loss = (1.0 - cos_sim[:, prompt_indices]).mean()
                elif self.args.use_random_selection:
                    # batchwise random selection
                    idx = torch.randint(self.e_pool_size, (1,)).repeat(cos_sim.size(0)).view(-1, 1)
                    loss = (1.0 - cos_sim[:,idx]).mean()
                    P_ = p[k_idx]
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).mean()
                    P_ = p[k_idx]
                    # count selected prompt when trianing
                    with torch.no_grad():
                        num = k_idx.view(-1).bincount(minlength=self.e_pool_size)
                        self.train_count += num
            else:
                if self.cls_to_prompt and label is not None: # pred 또는 GT
                    # select prompt to train by label
                    p_idx = label2prompt(label, cls_per_prompt=self.cls_per_prompt) # -> Torch.Tensor
                    p_idx = p_idx.view(-1, 1)
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    self.top_k_idx = top_k.indices
                    P_ = p[p_idx]
                    if not self.args.gt_key_value:
                        # count selected prompt when evaluating
                        with torch.no_grad():
                            num = p_idx.view(-1).bincount(minlength=self.e_pool_size)
                            self.eval_count += num
                else: # only pred
                    cos_sim = cos_sim[:, :self.task_count+1]
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    self.top_k_idx = k_idx
                    P_ = p[k_idx]
                    # count selected prompt when evaluating
                    with torch.no_grad():
                        num = k_idx.view(-1).bincount(minlength=self.e_pool_size)
                        self.eval_count += num
                
            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
            Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


class CodaBFPrompt(nn.Module):
    def __init__(self, args, emb_d, cls_per_prompt, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = args.n_splits
        self.cls_per_prompt = cls_per_prompt
        self._init_smart(emb_d)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d):

        # prompt basic param
        self.e_pool_size = self.args.e_prompt_pool_size
        self.e_p_length = self.args.e_prompt_length
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = self.args.ortho_mu

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def prompt_selection(self, x, s, f):
        sliced_x = []
        for start, end in zip(s, f):
            sliced_x.append(x[start:end])
        
        return torch.stack(sliced_x)

    def forward(self, x_querry, l, x_block, label=None, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            p_idx = label2prompt(label, cls_per_prompt=pt, e_pool_size=self.e_pool_size)
            end = (p_idx.max() + 1) * pt # TODO: exposed_classes를 사용하여 end를 계산하도록 수정
            
            # freeze/control past tasks
            if train:
                # instance-wise 계산: label에 따라 pt 단위로 prompt 선택
                p_idx = label2prompt(label, cls_per_prompt=pt, e_pool_size=self.e_pool_size)
                s = p_idx * pt
                f = (p_idx + 1) * pt
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = self.prompt_selection(K, s, f)
                    A = self.prompt_selection(A, s, f)
                    p = self.prompt_selection(p, s, f)
                    # K = K[s:f]
                    # A = A[s:f]
                    # p = p[s:f]
                # with attention and cosine sim
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                a_querry = torch.einsum('bd,bkd->bkd', x_querry, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,bkd->bk', q, n_K)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,bkld->bld', aq_k, p)

            else:
                K = K[0:end]
                A = A[0:end]
                p = p[0:end]

                # with attention and cosine sim
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


class L2P(BFPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = self.args.top_k
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

class OnePrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}',p)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = self.args.top_k
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt length
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
        else:
            p_return = None

        return p_return, 0, x_block

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

# label to prompt mapping function
def label2prompt(label: torch.Tensor, cls_per_prompt: int, e_pool_size: int = 10) -> torch.Tensor:
    """
    Map labels to prompt indices based on the given classes per prompt.
    
    Args:
        label (torch.Tensor): A tensor containing the labels.
        cls_per_prompt (int): Number of classes per prompt.
        
    Returns:
        torch.Tensor: Tensor containing the prompt indices.
    """
    # Compute initial prompt indices
    prompt_indices = label // cls_per_prompt

    # Ensure prompt indices are within the allowed range
    return torch.remainder(prompt_indices, e_pool_size)
    
# ETC


def ortho_penalty(t):
    # return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
    return ((t @t.T - torch.eye(t.shape[0]))**2).mean()


# supervised constrastive loss
def supervised_contrastive_loss(cos_sim, prompt_label, temperature=0.1):
    """
    Compute Supervised Contrastive Loss given cosine similarity and labels.

    Args:
        cos_sim (torch.Tensor): Cosine similarity matrix of shape (B, P),
                                where B is the number of queries, and P is the number of prompt keys.
        prompt_label (torch.Tensor): Tensor of shape (P,) containing the labels for prompt keys.
        temperature (float): Temperature parameter for scaling.

    Returns:
        torch.Tensor: Scalar tensor representing the supervised contrastive loss.
    """
    # Get the shape
    B, P = cos_sim.shape  # B: number of queries, P: number of prompt keys

    # Normalize cosine similarities using the temperature
    cos_sim = cos_sim / temperature

    # Create a mask for positive keys (same label as query)
    # Adjust mask to match (B, P)
    mask = prompt_label.unsqueeze(0) == prompt_label.unsqueeze(1)  # Shape: (P, P)
    mask = mask.float()  # Shape: (P, P)

    # Expand mask to align with query dimensions (B, P)
    # Note: The prompt_label must map to queries correctly
    mask = mask.repeat(B, 1)[:, :P]  # Adjust mask to match (B, P)

    # Compute the supervised contrastive loss
    exp_cos_sim = torch.exp(cos_sim)  # Exponentiated cosine similarity (B, P)
    numerator = exp_cos_sim * mask  # Mask out negatives
    denominator = exp_cos_sim.sum(dim=1, keepdim=True)  # Sum over all keys

    # Avoid division by zero
    denominator = denominator + 1e-8

    # Compute the loss
    loss_per_query = -torch.log(numerator.sum(dim=1) / denominator.squeeze(1))  # Shape: (B,)
    loss = loss_per_query.mean()  # Average over all queries

    return loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.z`
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss