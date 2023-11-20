# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.buffer_utils import maybe_cuda, n_classes, ClassBalancedRandomSampling
from utils.mir_utils import get_grad_vector
from utils.aser_utils import compute_knn_sv


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class BufferRetrieve:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, args, buffer_size, device, net, n_tasks=None, mode='reservoir', retrieve='random'):
        assert mode in ('ring', 'reservoir')
        assert retrieve in ('random', 'mir', 'aser')
        self.args = args
        self.buffer_size = buffer_size
        self.device = device
        self.net = net
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        if retrieve == 'mir':
            assert self.args.subsample is not None
        if retrieve == 'aser':
            self.n_smp_cls = int(self.args.n_smp_cls)
            self.out_dim = n_classes[self.args.dataset]
            self.is_aser_upt = False
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple
    
    ### MIR    
    def get_mir_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Maximally Interfered Retrieval samples a batch of size items.
        """
        sub_x, sub_y = self.get_data(size=self.args.subsample, transform=transform)
        grad_dims = []
        for param in self.net.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(self.net.parameters, grad_dims)
        net_temp = self.get_future_step_parameters(self.net, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = self.net.forward(sub_x)
                logits_post = net_temp.forward(sub_x)
                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                scores = post_loss - pre_loss
                big_ind = scores.sort(descending=True)[1][:size] # mem_mini_batch
            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y
    
    def get_future_step_parameters(self, net, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_net = deepcopy(net)
        self.overwrite_grad(new_net.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data = param.data - self.args.lr * param.grad.data
        return new_net
    
    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
    ### MIR End
    
    ### ASER
    def get_aser_data(self, cur_x, cur_y, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        model = self.net

        if self.num_seen_examples <= self.buffer_size:
            # Use random retrieval until buffer is filled
            ret_x, ret_y = self.get_data(size=size, transform=transform) # mem_mini_batch
        else:
            # Use ASER retrieval if buffer is filled
            # cur_x, cur_y = kwargs['x'], kwargs['y']
            buffer_x, buffer_y = self.get_all_data(transform=transform)
            ret_x, ret_y = self._retrieve_by_knn_sv(model, buffer_x, buffer_y, cur_x, cur_y, size) # mem_mini_batch
        return ret_x, ret_y
    
    def _retrieve_by_knn_sv(self, model, buffer_x, buffer_y, cur_x, cur_y, num_retrieve):
        """
            Retrieves data instances with top-N Shapley Values from candidate set.
                Args:
                    model (object): neural network.
                    buffer_x (tensor): data buffer.
                    buffer_y (tensor): label buffer.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                    num_retrieve (int): number of data instances to be retrieved.
                Returns
                    ret_x (tensor): retrieved data tensor.
                    ret_y (tensor): retrieved label tensor.
        """
        cur_x = maybe_cuda(cur_x)
        cur_y = maybe_cuda(cur_y)

        # Reset and update ClassBalancedRandomSampling cache if ASER update is not enabled
        if not self.is_aser_upt:
            ClassBalancedRandomSampling.update_cache(buffer_y, self.out_dim)

        # Get candidate data for retrieval (i.e., cand <- class balanced subsamples from memory)
        cand_x, cand_y, cand_ind = \
            ClassBalancedRandomSampling.sample(buffer_x, buffer_y, self.n_smp_cls, device=self.device)

        # Type 1 - Adversarial SV
        # Get evaluation data for type 1 (i.e., eval <- current input)
        eval_adv_x, eval_adv_y = cur_x, cur_y
        # Compute adversarial Shapley value of candidate data
        # (i.e., sv wrt current input)
        sv_matrix_adv = compute_knn_sv(model, eval_adv_x, eval_adv_y, cand_x, cand_y, self.args.k, device=self.device)

        if self.args.aser_type != "neg_sv":
            # Type 2 - Cooperative SV
            # Get evaluation data for type 2
            # (i.e., eval <- class balanced subsamples from memory excluding those already in candidate set)
            excl_indices = set(cand_ind.tolist())
            eval_coop_x, eval_coop_y, _ = \
                ClassBalancedRandomSampling.sample(buffer_x, buffer_y, self.n_smp_cls,
                                                   excl_indices=excl_indices, device=self.device)
            # Compute Shapley value
            sv_matrix_coop = \
                compute_knn_sv(model, eval_coop_x, eval_coop_y, cand_x, cand_y, self.args.k, device=self.device)
            if self.args.aser_type == "asv":
                # Use extremal SVs for computation
                sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
            else:
                # Use mean variation for aser_type == "asvm" or anything else
                sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
        else:
            # aser_type == "neg_sv"
            # No Type 1 - Cooperative SV; Use sum of Adversarial SV only
            sv = sv_matrix_adv.sum(0) * -1

        ret_ind = sv.argsort(descending=True)

        ret_x = cand_x[ret_ind][:num_retrieve]
        ret_y = cand_y[ret_ind][:num_retrieve]
        return ret_x, ret_y
    ### ASER End

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
