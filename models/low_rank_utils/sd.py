"""SDLoRA's current-task factors, historical scaling, and classifier update."""

from torch import nn
from torch.nn import functional as F


class SDLifecycle(nn.Module):
    @staticmethod
    def freeze(network, task):
        keys = [f'classifier_pool.{task}', f'lora_A_k.{task}', f'lora_A_v.{task}',
                f'lora_B_k.{task}', f'lora_B_v.{task}', 'scaling_factor']
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(any(key in name for key in keys))

    def begin_task(self, model, dataset):
        pass

    def observe(self, model, inputs, labels):
        logits = model.net(inputs)['logits']
        loss = F.cross_entropy(logits, labels)
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        return loss.item()

    def end_task(self, model, dataset):
        pass
