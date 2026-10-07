"""InfLoRA's SVD initialization and DualGPM gradient subspaces."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.low_rank_utils.continual import TensorList


def attention_modules(network):
    return (module for module in network.modules()
            if hasattr(module, 'cur_matrix') and hasattr(module, 'lora_A_k'))


class InfLifecycle(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_list = TensorList()
        self.register_buffer('project_codes', torch.empty(0, dtype=torch.long))

    @property
    def project_type(self):
        return ['remove' if int(code) == 0 else 'retain' for code in self.project_codes]

    @staticmethod
    def freeze(network, task):
        keys = [f'classifier_pool.{task}', f'lora_B_k.{task}', f'lora_B_v.{task}']
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(any(key in name for key in keys))

    @torch.no_grad()
    def begin_task(self, model, dataset):
        self._gather(model, dataset.train_loader)
        for index, module in enumerate(attention_modules(model.net)):
            matrix = module.cur_matrix
            if model.current_task:
                basis = self.feature_list[index].cpu().numpy()
                projection = torch.Tensor(np.dot(basis, basis.transpose())).to(matrix.device)
                if self.project_type[index] == 'remove':
                    matrix = matrix - torch.mm(projection, matrix)
                else:
                    matrix = torch.mm(projection, matrix)
                u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
            else:
                u, _, _ = torch.linalg.svd(matrix)
            value = u[:, :module.rank].T / math.sqrt(3)
            module.lora_A_k[model.current_task].weight.copy_(value)
            module.lora_A_v[model.current_task].weight.copy_(value)
            module.cur_matrix.zero_()
            module.n_cur_matrix = 0

    @staticmethod
    def _gather(model, loader):
        for batch in loader:
            model.net(batch[0].to(model.device), get_cur_feat=True)

    def observe(self, model, inputs, labels):
        logits = model.net(inputs)['logits']
        loss = F.cross_entropy(logits, labels)
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        return loss.item()

    @torch.no_grad()
    def end_task(self, model, dataset):
        self._gather(model, dataset.train_loader)
        matrices = []
        for module in attention_modules(model.net):
            matrices.append(module.cur_matrix.detach().cpu().numpy().copy())
            module.cur_matrix.zero_()
            module.n_cur_matrix = 0
        threshold = ((model.args.lora_lame - model.args.lora_lamb)
                     * model.current_task / model.n_tasks + model.args.lora_lamb)
        self.update_dualgpm(matrices, threshold)

    def update_dualgpm(self, matrices, threshold):
        """Preserve the source NumPy SVD, threshold counts, and complements."""
        features = [feature.cpu().numpy() for feature in self.feature_list]
        types = self.project_type
        if not features:
            for activation in matrices:
                u, s, _ = np.linalg.svd(activation, full_matrices=False)
                ratio = s ** 2 / (s ** 2).sum()
                rank = np.sum(np.cumsum(ratio) < threshold)
                features.append(u[:, :max(rank, 1)])
                types.append('remove' if rank < activation.shape[0] / 2 else 'retain')
        else:
            for index, activation in enumerate(matrices):
                _, s1, _ = np.linalg.svd(activation, full_matrices=False)
                total = (s1 ** 2).sum()
                basis = features[index]
                projected = np.dot(np.dot(basis, basis.transpose()), activation)
                residual = activation - projected if types[index] == 'remove' else projected
                u, s, _ = np.linalg.svd(residual, full_matrices=False)
                ratio = s ** 2 / total
                projected_energy = (s ** 2).sum()
                if types[index] == 'remove':
                    accumulated = (total - projected_energy) / total
                    rank = 0
                    for value in ratio:
                        if accumulated < threshold:
                            accumulated += value
                            rank += 1
                        else:
                            break
                    if rank:
                        updated = np.hstack((basis, u[:, :rank]))
                        features[index] = updated[:, :updated.shape[0]]
                else:
                    accumulated = projected_energy / total
                    rank = 0
                    for value in ratio:
                        if accumulated >= 1 - threshold:
                            accumulated -= value
                            rank += 1
                        else:
                            break
                    if rank:
                        remaining = basis - np.dot(np.dot(u[:, :rank], u[:, :rank].transpose()), basis)
                        updated, _, _ = np.linalg.svd(remaining)
                        features[index] = updated[:, :basis.shape[1] - rank]
        for index, (feature, project_type) in enumerate(zip(features, types)):
            if project_type == 'remove' and feature.shape[1] > feature.shape[0] / 2:
                u, _, _ = np.linalg.svd(feature)
                features[index] = u[:, feature.shape[1]:]
                types[index] = 'retain'
            elif project_type == 'retain':
                assert feature.shape[1] <= feature.shape[0] / 2
            value = torch.from_numpy(features[index].copy()).to(self.project_codes.device)
            if index < len(self.feature_list):
                self.feature_list[index] = value
            else:
                self.feature_list.append(value)
        self.project_codes = torch.tensor([0 if value == 'remove' else 1 for value in types],
                                          dtype=torch.long, device=self.project_codes.device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'project_codes'
        if key in state_dict:
            self.project_codes = torch.empty_like(state_dict[key], device=self.project_codes.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
