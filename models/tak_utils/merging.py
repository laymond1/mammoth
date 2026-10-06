import torch
from abc import ABC, abstractmethod

from typing import Dict
from models.tak_utils.ties_merging import ties_merging
from tqdm.auto import tqdm


def get_merging_function(command_args, device):
    if command_args.merging == 'ta':
        return TaskArithmetic(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'dare':
        return DARE(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'iso':
        return ISO(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'ties':
        return TIES(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'tsv':
        return TSV(device, alpha=command_args.alpha_merging)
    else:
        raise ValueError


class AbstractMerging(ABC):

    @abstractmethod
    def merge(self):
        raise NotImplementedError
    
    @abstractmethod
    def add(self, param_dict: Dict):
        raise NotImplementedError
    
    def set_alpha(self, alpha: float):
        self.alpha = alpha


class TaskArithmetic(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict|None = None
        self.scaled_sum: Dict|None = None

    @torch.no_grad()
    def merge(self, names=None):
        assert self.scaled_sum and self._running_sum
        alpha = (1/self.num_tasks) * self.alpha
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k in self._running_sum.keys():
            self.scaled_sum[k].copy_(self._running_sum[k])
            self.scaled_sum[k].mul_(alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            self._running_sum[k].add_(v)
        self.num_tasks += 1

class DARE(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0, p: float = 0.7):
        self.device = device
        self.alpha = alpha
        self.p = p
        self.num_tasks = 0
        self._running_sum: Dict|None = None
        self.scaled_sum: Dict|None = None

    def randbin(self, M, N):
        return torch.randint(2, size=(M, N), dtype=torch.float32).\
            bernoulli(1 - self.p).to(self.device)

    @torch.no_grad()
    def merge(self, names=None):
        assert self.scaled_sum and self._running_sum
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k, v in self._running_sum.items():
            self.scaled_sum[k].copy_(v)
            if len(v.shape) != 2:
                self.scaled_sum[k].mul_(self.alpha * (1/self.num_tasks))
            else:
                self.scaled_sum[k].mul_(self.alpha * (1/self.num_tasks))
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            if len(v.shape) != 2:
                self._running_sum[k].add_(v)
            else:
                mask_ = self.randbin(v.shape[0], v.shape[1])
                self._running_sum[k].add_(v * mask_ * (1/(1-self.p)))
        self.num_tasks += 1

class ISO(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict|None = None
        self.scaled_sum: Dict|None = None

    @torch.no_grad()
    def merge(self, names=None):
        assert self.scaled_sum and self._running_sum
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k, v in tqdm(self._running_sum.items(),
                         desc="Computing SVD for ISO",
                         total=len(self._running_sum)):
            self.scaled_sum[k].copy_(v)
            self.scaled_sum[k].div_(self.num_tasks)
            if len(v.shape) == 2:
                U, S, V = torch.linalg.svd(self.scaled_sum[k].to(torch.double),
                                           full_matrices=False)
                self.scaled_sum[k].copy_((U @ V).to(self.scaled_sum[k].dtype))
                self.scaled_sum[k].mul_(S.mean())
            self.scaled_sum[k].mul_(self.alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            self._running_sum[k].add_(v)
        self.num_tasks += 1

class TIES(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict|None = None
        self._separated_task_vectors: Dict|None = None
        self.merged_model: Dict|None = None

    def apply_ta(self, v):
        if len(v.shape) == 2:
            return False
        return True

    @torch.no_grad()
    def merge(self, names=None):
        assert self._separated_task_vectors and self._running_sum and self.merged_model
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].mul_(self.alpha / self.num_tasks)
        for k, v in self._separated_task_vectors.items():
            merged_tv, _, _ = ties_merging(v)
            self.merged_model[k].copy_((self.alpha / self.num_tasks) * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items() if self.apply_ta(v)}
            self._separated_task_vectors = {k: [] for k, v in param_dict.items() if not self.apply_ta(v)}

        assert self._separated_task_vectors
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1

class TSV(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict|None = None
        self._separated_task_vectors: Dict|None = None
        self.merged_model: Dict|None = None

    def apply_ta(self, v):
        if len(v.shape) == 2:
            return False
        return True

    @torch.no_grad()
    def get_tsv_delta_w(self, ftms_task_dirs):
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, v = torch.linalg.svd(vec.to(torch.float64), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[ # pyright: ignore[reportPossiblyUnboundVariable]
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[ # pyright: ignore[reportPossiblyUnboundVariable]
                :reduced_index_s
            ]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[ # pyright: ignore[reportPossiblyUnboundVariable]
                :reduced_index_s, :
            ]
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False) # pyright: ignore[reportPossiblyUnboundVariable]
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False) # pyright: ignore[reportPossiblyUnboundVariable]

        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(ftms_task_dirs[0]) # pyright: ignore[reportPossiblyUnboundVariable]

    @torch.no_grad()
    def merge(self, names=None):
        assert self.merged_model and self._running_sum and self._separated_task_vectors
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].div_(self.num_tasks)
            self.merged_model[k].mul_(self.alpha)
        for k, v in self._separated_task_vectors.items():
            merged_tv = self.get_tsv_delta_w(v)
            merged_tv = merged_tv.type_as(v[0]) if \
                hasattr(merged_tv, 'type_as') else merged_tv
            self.merged_model[k].copy_(self.alpha * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items() if self.apply_ta(v)}
            self._separated_task_vectors = {k: [] for k, v in param_dict.items() if not self.apply_ta(v)}
        assert self._separated_task_vectors
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1
        