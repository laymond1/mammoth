"""CLLoRA's two updates, gradient reassignment, and prototype lifecycle."""

import torch
from torch import nn
from torch.nn import functional as F

from models.low_rank_utils.continual import SourceLoader


def kd_loss(prediction, teacher, temperature):
    prediction = torch.log_softmax(prediction / temperature, dim=1)
    teacher = torch.softmax(teacher / temperature, dim=1)
    return -(teacher * prediction).sum() / prediction.shape[0]


def orthogonality_loss(previous, current, epsilon=1e-8):
    current = current.flatten() / (torch.norm(current.flatten()) + epsilon)
    loss = 0.
    for weights in previous:
        weights = weights.flatten() / (torch.norm(weights.flatten()) + epsilon)
        loss = loss + torch.abs(torch.sum(weights * current))
    return loss / len(previous) if len(previous) else loss


def iter_attn_lora_b(network, position, projection, use_new):
    prefix = f'image_encoder.blocks.{position}.attn'
    key = f'lora_{projection}' if use_new else f'lora_{projection}_old'
    for name, parameter in network.named_parameters():
        if name.startswith(prefix) and key in name and name.endswith('lora_B.weight'):
            yield parameter


class CLLifecycle(nn.Module):
    @staticmethod
    def freeze(network, task):
        keys = ['lora_q.lora_B', 'lora_v.lora_B', f'lora_q.{task}.lora',
                f'lora_v.{task}.lora', 'proxy_fc']
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(any(key in name for key in keys)
                                     or name.endswith(f'block_weights.{task}'))

    def begin_task(self, model, dataset):
        pass

    def observe(self, model, inputs, labels):
        # The source computes CE before applying KD, then backpropagates that
        # original graph. Clone saved values on mutation to retain its exact
        # gradient without autograd's in-place parameter version error.
        with torch.autograd.graph.allow_mutation_on_saved_tensors():
            logits = model.net(inputs, model.current_task)['logits']
            loss = F.cross_entropy(logits, labels)
            if model.current_task:
                prediction, teacher = model.net.forward_kd(inputs)
                distillation = model.args.lora_kd_ratio * kd_loss(prediction, teacher, model.args.lora_temperature)
                model.opt.zero_grad()
                distillation.backward()
                with torch.no_grad():
                    for position in model.args.lora_shared_pos:
                        for index, enabled in enumerate(model.args.lora_msa):
                            if not enabled:
                                continue
                            projection = ('q', 'k', 'v')[index]
                            old_b = next(iter_attn_lora_b(model.net, position, projection, False)).detach()
                            scale = torch.norm(old_b, dim=1)
                            scale = len(scale) * scale / torch.sum(scale)
                            new_b = next(iter_attn_lora_b(model.net, position, projection, True))
                            if new_b.grad is not None:
                                new_b.grad.mul_(scale.unsqueeze(1))
                model.opt.step()
                weights = model.net.image_encoder.block_weights
                loss = loss + 0.0001 * orthogonality_loss(weights[:model.current_task], weights[model.current_task])
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            return loss.item()

    @torch.no_grad()
    def end_task(self, model, dataset):
        model.net.replace_fc(SourceLoader(model._prototype_loader(dataset)))
        model.net.save_old_shared_lora()
