import torch

from utils.buffer_utils import maybe_cuda


def get_grad_vector(pp, grad_dims):
    """
        gather the gradients in one vector
    """
    grads = maybe_cuda(torch.Tensor(sum(grad_dims)))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads