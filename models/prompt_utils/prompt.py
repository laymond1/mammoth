# This code is a reimplementation based on the OVOR methodology.
# This code has been modified for online continual learning by Wonseon Lim.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPrompt(nn.Module):
    def __init__(self, args, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self._init_smart(emb_d)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}',p)
            if args.same_key_value and e == self.e_layers[-1]:
                k = tensor_prompt(self.e_pool_size, self.key_d)
                setattr(self, f'e_k',k)
            else:
                k = tensor_prompt(self.e_pool_size, self.key_d)
                setattr(self, f'e_k_{e}',k)

        # init
        self.register_buffer('train_count', torch.zeros(self.e_pool_size))
        self.register_buffer('eval_count', torch.zeros(self.e_pool_size))

    def _init_smart(self, emb_d):
        
        self.top_k = self.args.top_k
        if self.args.online_scenario in ['online-stand-cil', 'online-cil']: 
            self.task_id_bootstrap = True
        else:
            self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = self.args.g_prompt_layer_idx
        self.e_layers = self.args.e_prompt_layer_idx

        # prompt pool size
        self.g_p_length = self.args.g_prompt_length
        self.e_p_length = self.args.e_prompt_length
        self.e_pool_size = self.args.e_prompt_pool_size
    
    def process_task_count(self):
        self.task_count += 1

        # transfer prompts
        for e in self.e_layers:
            p = getattr(self,f'e_p_{e}')
            p = self.transfer_prompt(p, self.task_count-1, self.task_count)
            setattr(self, f'e_p_{e}',p)

    def transfer_prompt(self, prev_p, prev_task_id, new_task_id):
        noise = torch.randn_like(prev_p[new_task_id], device=prev_p.device)
        new_p = torch.zeros_like(prev_p, device=prev_p.device)
        new_p[:new_task_id] = prev_p[:new_task_id].clone()
        new_p[new_task_id] = prev_p[prev_task_id].clone() + noise
        return torch.nn.Parameter(new_p, requires_grad=True)


    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            if self.args.same_key_value:
                K = getattr(self,f'e_k')
            else:
                K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dualprompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, self.task_count]).mean()
                    P_ = p[self.task_count].expand(len(x_querry),-1,-1)
                    # count selected prompt when trianing
                    with torch.no_grad():
                        num = torch.zeros(self.e_pool_size, device=self.train_count.device)
                        num[self.task_count] = B
                        self.train_count += num
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
                # dualprompt's activated prompts
                if self.task_id_bootstrap:
                    cos_sim = cos_sim[:, :self.task_count+1]
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                # count selected prompt when evaluating
                with torch.no_grad():
                    num = k_idx.view(-1).bincount(minlength=self.e_pool_size)
                    self.eval_count += num
            
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
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


class L2P(DualPrompt):
    def __init__(self, args, emb_d, prompt_param, key_dim=768):
        super().__init__(args, emb_d, prompt_param, key_dim)

    def _init_smart(self, emb_d):
        self.top_k = self.args.top_k
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if not self.args.shallow: 
            self.e_layers = [0,1,2,3,4] # deep
        else:
            self.e_layers = [0] # shallow

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = self.args.e_prompt_length
        self.e_pool_size = self.args.e_prompt_pool_size


class CodaPrompt(nn.Module):
    def __init__(self, args, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        if args.n_splits is not None:
            self.n_tasks = args.n_splits
        else:
            self.n_tasks = args.n_tasks
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
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_a_{e}',a)
            if args.same_key_value and e == self.e_layers[-1]:
                k = tensor_prompt(self.e_pool_size, self.key_d)
                k = self.gram_schmidt(k)
                setattr(self, f'e_k',k)
            else:
                k = tensor_prompt(self.e_pool_size, self.key_d)
                k = self.gram_schmidt(k)
                setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d):

        # prompt basic param
        self.e_pool_size = self.args.e_prompt_pool_size
        self.e_p_length = self.args.e_prompt_length
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = self.args.ortho_mu

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        if self.args.n_splits is None:
            for e in self.e_layers:
                K = getattr(self,f'e_k_{e}')
                A = getattr(self,f'e_a_{e}')
                P = getattr(self,f'e_p_{e}')
                k = self.gram_schmidt(K)
                a = self.gram_schmidt(A)
                p = self.gram_schmidt(P)
                setattr(self, f'e_p_{e}',p)
                setattr(self, f'e_k_{e}',k)
                setattr(self, f'e_a_{e}',a)
        else:
            self.task_count = 0

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

    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            if self.args.same_key_value:
                K = getattr(self,f'e_k')
            else:
                K = getattr(self,f'e_k_{l}') # 0 based indexing here
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks)) # E-prompt pool size
            s = int(self.task_count * pt) # 0
            f = int((self.task_count + 1) * pt) # pt
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

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


class MVPPrompt(nn.Module):
    def __init__(self, args, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.use_mask = args.use_mask
        self.emb_d = emb_d
        self.key_d = key_dim
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

        # init
        self.features = torch.empty(0)
        self.mask = nn.Parameter(torch.zeros(self.e_pool_size, self.args.num_classes) - 1)
        self.register_buffer('train_count', torch.zeros(self.e_pool_size))
        self.register_buffer('eval_count', torch.zeros(self.e_pool_size))

    def _init_smart(self, emb_d):
        
        self.top_k = self.args.top_k

        # prompt locations
        self.g_layers = self.args.g_prompt_layer_idx
        self.e_layers = self.args.e_prompt_layer_idx

        # prompt pool size
        self.g_p_length = self.args.g_prompt_length
        self.e_p_length = self.args.e_prompt_length
        self.e_pool_size = self.args.e_prompt_pool_size

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False):
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k') # sinle key for all e_layers
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            # contrastive visual prompt tuning loss
            distance = 1 - cos_sim
            if self.args.use_contrastiv:
                mass = (self.train_count + 1)
            else:
                mass = 1.0
            scaled_distance = distance * mass
            top_k = torch.topk(scaled_distance, self.top_k, dim=1, largest=False)
            k_idx = top_k.indices
            distance = distance[torch.arange(k_idx.size(0), device=k_idx.device).unsqueeze(1).repeat(1,self.top_k), k_idx].squeeze().clone()
            P_ = p[k_idx] # e_prompts
            
            if self.args.use_contrastiv:
                key_wise_distance = 1 - F.cosine_similarity(n_K.unsqueeze(1), n_K.detach(), dim=-1)
                loss = -((key_wise_distance[k_idx] / mass[k_idx]).exp().mean() / ((distance / mass[k_idx]).exp().mean() + (key_wise_distance[k_idx] / mass[k_idx]).exp().mean()) + 1e-6).log()
            else:
                loss = (1.0 - cos_sim[:,k_idx]).mean()

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
            Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))

            # count selected prompt when trianing
            if train:
                with torch.no_grad():
                    num = k_idx.view(-1).bincount(minlength=self.e_pool_size)
                    self.train_count += num
            else:
                # count selected prompt when evaluating
                with torch.no_grad():
                    num = k_idx.view(-1).bincount(minlength=self.e_pool_size)
                    self.eval_count += num

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
        

class OnePrompt(nn.Module):
    def __init__(self, args, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self._init_smart(emb_d)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}',p)

    def _init_smart(self, emb_d):
        
        # self.top_k = self.args.top_k

        # prompt locations
        self.g_layers = self.args.g_prompt_layer_idx
        self.e_layers = self.args.e_prompt_layer_idx

        # prompt length
        self.g_p_length = self.args.g_prompt_length
        self.e_p_length = self.args.e_prompt_length

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

# coda-prompt
def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).to(t.device))**2).mean()
    # return ((t @ t.T - torch.eye(t.shape[0]))**2).mean()

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
    
