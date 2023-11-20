import torch
from collections import defaultdict


n_classes = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 100,
}


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what

def mini_batch_deep_features(model, total_x, num):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
            Returns
                deep_features (tensor): deep feature representation of data tensor.
    """
    is_train = False
    if model.training:
        is_train = True
        model.eval()
    if hasattr(model, "_features"):
        model_has_feature_extractor = True
    else:
        model_has_feature_extractor = False
        # delete the last fully connected layer
        modules = list(model.children())[:-1]
        # make feature extractor
        model_features = torch.nn.Sequential(*modules)

    with torch.no_grad():
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features_list = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]

            if model_has_feature_extractor:
                batch_deep_features_ = model(batch_x, returnt='features')
            else:
                batch_deep_features_ = torch.squeeze(model_features(batch_x))

            deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
            sid = eid
        if num_itr == 1:
            deep_features_ = deep_features_list[0]
        else:
            deep_features_ = torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    return deep_features_


def euclidean_distance(u, v):
    euclidean_distance_ = (u - v).pow(2).sum(1)
    return euclidean_distance_


def ohe_label(label_tensor, dim, device="cpu"):
    # Returns one-hot-encoding of input label tensor
    n_labels = label_tensor.size(0)
    zero_tensor = torch.zeros((n_labels, dim), device=device, dtype=torch.long)
    return zero_tensor.scatter_(1, label_tensor.reshape((n_labels, 1)), 1)


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]


class ClassBalancedRandomSampling:
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}
    class_index_cache = None
    class_num_cache = None

    @classmethod
    def sample(cls, buffer_x, buffer_y, n_smp_cls, excl_indices=None, device="cpu"):
        """
            Take same number of random samples from each class from buffer.
                Args:
                    buffer_x (tensor): data buffer.
                    buffer_y (tensor): label buffer.
                    n_smp_cls (int): number of samples to take from each class.
                    excl_indices (set): indices of buffered instances to be excluded from sampling.
                    device (str): device for tensor allocation.
                Returns
                    x (tensor): class balanced random sample data tensor.
                    y (tensor): class balanced random sample label tensor.
                    sample_ind (tensor): class balanced random sample index tensor.
        """
        if excl_indices is None:
            excl_indices = set()

        # Get indices for class balanced random samples
        # cls_ind_cache = class_index_tensor_list_cache(buffer_y, num_class, excl_indices, device=device)

        sample_ind = torch.tensor([], device=device, dtype=torch.long)

        # Use cache to retrieve indices belonging to each class in buffer
        for ind_set in cls.class_index_cache.values():
            if ind_set:
                # Exclude some indices
                valid_ind = ind_set - excl_indices
                # Auxiliary indices for permutation
                perm_ind = torch.randperm(len(valid_ind), device=device)
                # Apply permutation, and select indices
                ind = torch.tensor(list(valid_ind), device=device, dtype=torch.long)[perm_ind][:n_smp_cls]
                sample_ind = torch.cat((sample_ind, ind))

        x = buffer_x[sample_ind]
        y = buffer_y[sample_ind]

        x = maybe_cuda(x)
        y = maybe_cuda(y)

        return x, y, sample_ind

    @classmethod
    def update_cache(cls, buffer_y, num_class, new_y=None, ind=None, device="cpu"):
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
        if cls.class_index_cache is None:
            # Initialize caches
            cls.class_index_cache = defaultdict(set)
            cls.class_num_cache = torch.zeros(num_class, dtype=torch.long, device=device)

        if new_y is not None:
            # If ASER update is being used, keep updating existing caches
            # Get labels of memory samples to be replaced
            orig_y = buffer_y[ind]
            # Update caches
            for i, ny, oy in zip(ind, new_y, orig_y):
                oy_int = oy.item()
                ny_int = ny.item()
                i_int = i.item()
                # Update dictionary according to new class label of index i
                if oy_int in cls.class_index_cache and i_int in cls.class_index_cache[oy_int]:
                    cls.class_index_cache[oy_int].remove(i_int)
                    cls.class_num_cache[oy_int] -= 1
                cls.class_index_cache[ny_int].add(i_int)
                cls.class_num_cache[ny_int] += 1
        else:
            # If only ASER retrieve is being used, reset cache and update it based on buffer
            cls_ind_cache = defaultdict(set)
            for i, c in enumerate(buffer_y):
                cls_ind_cache[c.item()].add(i)
            cls.class_index_cache = cls_ind_cache