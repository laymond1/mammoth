from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer


class SCR_Buffer(Buffer):
    """
    SCR Buffer is taken from 
    Online Continual Learning in Image Classification: An Empirical Survey code
    : [https://github.com/RaptorMai/online-continual-learning]
    """
    def __init__(self, args, buffer_size, device, n_tasks=None, mode='reservoir'):
        super(SCR_Buffer, self).__init__(buffer_size, device, n_tasks, mode)
        self.args = args
        self.current_index = 0
        # self.n_seen_so_far = 0 : equal self.num_seen_examples
        
        self.buffer_tracker = BufferClassTracker(num_class=get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK,
                                                 device=device)
        
    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
            
        batch_size = examples.shape[0]
        # add whatever still fits in the buffer
        place_left = max(0, self.examples.shape[0] - self.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            self.examples[self.current_index: self.current_index+offset].data.copy_(examples[:offset])
            if labels is not None:
                self.labels[self.current_index: self.current_index+offset].data.copy_(labels[:offset])
            if logits is not None:
                self.logits[self.current_index: self.current_index+offset].data.copy_(logits[:offset])
            if task_labels is not None:
                self.task_labels[self.current_index: self.current_index+offset].data.copy_(task_labels[:offset])

            self.current_index += offset
            self.num_seen_examples += offset
            
            # everything was added
            if offset == examples.size(0):
                filled_idx = list(range(self.current_index - offset, self.current_index, ))
                if self.buffer_tracker:
                    self.buffer_tracker.update_cache(self.labels, labels[:offset], filled_idx)
                return filled_idx
            
        #TODO: the buffer tracker will have bug when the mem size can't be divided by batch size
        
        # remove what is already in the buffer
        examples, labels = examples[place_left:], labels[place_left:]

        indices = torch.FloatTensor(examples.size(0)).to(examples.device).uniform_(0, self.num_seen_examples).long()
        valid_indices = (indices < self.examples.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.num_seen_examples += examples.size(0)

        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < self.examples.size(0)
        assert idx_buffer.max() < self.labels.size(0)
        # assert idx_buffer.max() < self.task_labels.size(0)

        assert idx_new_data.max() < examples.size(0)
        assert idx_new_data.max() < labels.size(0)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = labels[list(idx_map.values())]
        if self.buffer_tracker:
            self.buffer_tracker.update_cache(self.labels, replace_y, list(idx_map.keys()))
        # perform overwrite op
        self.examples[list(idx_map.keys())] = examples[list(idx_map.values())]
        self.labels[list(idx_map.keys())] = replace_y
        return list(idx_map.keys())
    
    def get_data(self, size: int, transform: nn.Module = None, excl_indices=None, return_index=False):
        """
        Random retrieve sample.
        """
        filled_indices = np.arange(self.current_index)
        if excl_indices is not None:
            excl_indices = list(excl_indices)
        else:
            excl_indices = []
        valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
        size = min(size, valid_indices.shape[0])
        indices = torch.from_numpy(np.random.choice(valid_indices, size, replace=False)).long()

        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[indices]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[indices],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(indices).to(self.device), ) + ret_tuple
    

class BufferClassTracker(object):
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}

    def __init__(self, num_class, device="cpu"):
        super().__init__()
        # Initialize caches
        self.class_index_cache = defaultdict(set)
        self.class_num_cache = np.zeros(num_class)

    def update_cache(self, buffer_y, new_y=None, ind=None, ):
        """
            Collect indices of buffered data from each class in set.
            Update class_index_cache with list of such sets.
                Args:
                    buffer_y (tensor): label buffer.
                    num_class (int): total number of unique class labels.
                    new_y (tensor): label tensor for replacing memory samples at ind in buffer.
                    ind (tensor): indices of memory samples to be updated.
                    device (str): device for tensor allocation.
        """

        # Get labels of memory samples to be replaced
        orig_y = buffer_y[ind]
        # Update caches
        for i, ny, oy in zip(ind, new_y, orig_y):
            oy_int = oy.item()
            ny_int = ny.item()
            # Update dictionary according to new class label of index i
            if oy_int in self.class_index_cache and i in self.class_index_cache[oy_int]:
                self.class_index_cache[oy_int].remove(i)
                self.class_num_cache[oy_int] -= 1

            self.class_index_cache[ny_int].add(i)
            self.class_num_cache[ny_int] += 1

    def check_tracker(self):
        print(self.class_num_cache.sum())
        print(len([k for i in self.class_index_cache.values() for k in i]))
