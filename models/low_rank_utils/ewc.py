"""EWCLoRA: empirical Fisher in weight space, followed by merge/reset."""

import torch
from torch import nn
from torch.nn import functional as F

from models.low_rank_utils.continual import TensorList


class EWCLifecycle(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega_W = TensorList()
        self.register_buffer('count_updates', torch.tensor(0, dtype=torch.long))
        self.register_buffer('pending_merge', torch.tensor(False))

    @staticmethod
    def freeze(network, task):
        keys = [f'classifier_pool.{task}', 'lora_new_A_k', 'lora_new_A_v',
                'lora_new_B_k', 'lora_new_B_v']
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(any(key in name for key in keys))

    def begin_task(self, model, dataset):
        # A checkpoint taken between end_task and evaluation keeps the exact
        # pre-merge prediction state. Finish its source after_task before a new
        # task starts, even if the caller did not run the evaluation hook.
        self.finalize_task(model)

    def observe(self, model, inputs, labels):
        logits = model.net(inputs, use_new=True)['logits']
        loss = F.cross_entropy(logits, labels)
        if int(self.count_updates):
            new_a = (p for p in model.net.parameters() if getattr(p, '_is_new_a', False))
            new_b = (p for p in model.net.parameters() if getattr(p, '_is_new_b', False))
            ewc_loss = 0.
            for index, (a, b) in enumerate(zip(new_a, new_b)):
                delta = b @ a
                ewc_loss = ewc_loss + torch.sum(self.omega_W[index].float().to(delta.device) * delta.square())
            loss = loss + model.args.lora_ewc_lambda / 2. * ewc_loss
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        return loss.item()

    def end_task(self, model, dataset):
        fisher = compute_fisher(model.net, dataset.train_loader, model.n_past_classes, model.device)
        self.count_updates.add_(1)
        for index, matrix in enumerate(fisher):
            if index < len(self.omega_W):
                self.omega_W[index] = model.args.lora_gamma * self.omega_W[index].to(matrix.device) + matrix
            else:
                self.omega_W.append(matrix)
        self.pending_merge.fill_(True)

    def finalize_task(self, model):
        if bool(self.pending_merge):
            model.net.accumulate_and_reset_lora()
            self.pending_merge.fill_(False)


def compute_fisher(network, loader, class_offset, device):
    """Source batch-mean empirical Fisher, weighted by batch sample count."""
    network.eval()
    fisher = []
    for module in network.modules():
        for projection in ('k', 'v'):
            if hasattr(module, f'lora_new_A_{projection}') and hasattr(module, f'lora_new_B_{projection}'):
                a = getattr(module, f'lora_new_A_{projection}').weight
                b = getattr(module, f'lora_new_B_{projection}').weight
                fisher.append(torch.zeros_like(b @ a))
    samples = 0
    # end_task is also used by callers under no_grad; Fisher needs autograd.
    with torch.enable_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            network.zero_grad()
            logits = network(inputs, use_new=True, register_hook=True)['logits']
            F.cross_entropy(logits, labels - class_offset).backward()
            batch_size = inputs.size(0)
            samples += batch_size
            index = 0
            for module in network.modules():
                for projection in ('k', 'v'):
                    attribute = f'delta_w_{projection}_new_grad'
                    if hasattr(module, attribute):
                        gradient = getattr(module, attribute)
                        if gradient is not None:
                            fisher[index].add_(gradient.detach().square(), alpha=batch_size)
                        index += 1
    if samples == 0:
        raise ValueError('EWCLoRA needs task training samples to estimate its Fisher')
    return [matrix / samples for matrix in fisher]
