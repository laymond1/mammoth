"""PROL class keys and shared convolutional prefix generator.

This preserves the author's active indexing, including its batch-wide advanced
index assignment, local class indices and evaluation-time affine clamps. These
are source behavior, not corrections to the paper's equations.
"""

# Adapted from anwarmaxsum/PROL, promptL.py, bfff8418 (MIT, Copyright (c) 2025).
# The accompanying LICENSE retains the author's complete license notice.

import torch
from torch import nn
from torch.nn import functional as F


class ProlPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, num_tasks=10, num_classes=100,
                 embedding_key='cls', prompt_init='uniform', prompt_pool=True,
                 prompt_key=True, pool_size=None, top_k=1, top_k_l=1,
                 batchwise_prompt=False, prompt_key_init='uniform', num_layers=7,
                 use_prefix_tune_for_e_prompt=True, num_heads=12,
                 same_key_value=False, prompts_per_task=5):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in
                (length, embed_dim, num_tasks, num_classes, num_layers, num_heads))
                or num_classes % num_tasks or embed_dim % num_heads or top_k != 1 or top_k_l != 1
                or not prompt_pool or not prompt_key or not use_prefix_tune_for_e_prompt
                or embedding_key != 'cls' or prompt_key_init not in ('zero', 'uniform', 'ortho')
                or batchwise_prompt or same_key_value):
            raise ValueError('PROL supports its active CLS-query, independent K/V, top_k_l=1 prefix path')
        self.length, self.embed_dim = length, embed_dim
        self.prompt_pool, self.embedding_key, self.prompt_init = prompt_pool, embedding_key, prompt_init
        self.top_k, self.top_k_l, self.n_tasks = top_k, top_k_l, num_tasks
        self.batchwise_prompt, self.num_layers = batchwise_prompt, num_layers
        self.use_prefix_tune_for_e_prompt, self.num_heads = use_prefix_tune_for_e_prompt, num_heads
        self.same_key_value, self.use_prompt_embed_matcher = same_key_value, True
        self.prompts_per_task = prompts_per_task
        self.old_num_k = self.new_num_k = self.old_num_c = self.new_num_c = 0
        self.sigmoid = nn.Sigmoid()
        self.num_classes, self.numclasses_per_task = num_classes, num_classes // num_tasks
        self.kmax_list, self.lmax_list = [], []
        self.key_length, self.pool_size = embed_dim, num_tasks * prompts_per_task
        # Construction order matches LPrompt: all K convolutions, then V, then
        # affine parameters, then the class keys (including its discarded draw).
        self.k_comp_gen = self._generators()
        self.v_comp_gen = self._generators()
        affine_shape = (num_classes, length)
        self.scaler = nn.Parameter(torch.ones(affine_shape))
        self.shifter = nn.Parameter(torch.zeros(affine_shape))
        self.scaler_v = nn.Parameter(torch.ones(affine_shape))
        self.shifter_v = nn.Parameter(torch.zeros(affine_shape))
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(num_classes, embed_dim))
        else:
            self.prompt_key = nn.Parameter(torch.randn(num_classes, embed_dim))
            if prompt_key_init == 'uniform':
                nn.init.uniform_(self.prompt_key, -1, 1)
            else:
                nn.init.orthogonal_(self.prompt_key)

    def _generators(self):
        return nn.ModuleDict({str(layer): nn.ModuleDict({str(head): nn.ModuleList(
            nn.Conv1d(1, 1, 3, stride=1, padding=1) for _ in range(self.top_k_l))
            for head in range(self.num_heads)}) for layer in range(self.num_layers)})

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def process_new_task(self, old_num_k, new_num_k, new_desc_embed=None):
        if (type(old_num_k) is not int or type(new_num_k) is not int
                or old_num_k != len(self.lmax_list) or new_num_k != old_num_k + self.top_k_l
                or len(self.lmax_list) >= self.n_tasks or new_desc_embed is not None):
            raise ValueError('PROL requires sequential one-generator task transitions without descriptors')
        self.old_num_k, self.new_num_k = old_num_k, new_num_k
        if self.old_num_k == 0:
            self.kmax_list.append(new_num_k)
        self.old_num_c = self.new_num_c
        self.new_num_c += self.numclasses_per_task
        self.lmax_list.append(self.new_num_c)
        # The only generator is frozen at the beginning of the second task.
        if len(self.lmax_list) == 2:
            for generators in (self.k_comp_gen, self.v_comp_gen):
                for layer in generators.values():
                    for head in layer.values():
                        for index in range(self.old_num_k):
                            head[index].requires_grad_(False)

    def forward(self, x_embed, y, task_id=-1, prompt_mask=None, layer_num=-1, cls_features=None):
        if not self.lmax_list or cls_features is None or not 0 <= layer_num < self.num_layers:
            raise ValueError('PROL prefixes require task state, a frozen CLS query and a valid layer')
        if self.training and (task_id != len(self.lmax_list) - 1 or y is None):
            raise ValueError('PROL training requires current-task labels and its task index')
        output = {}
        # This first normalization is recomputed below in the author path. Keep
        # its active work for faithful compute accounting despite the overwrite.
        prompt_key_norm = self.l2_normalize(self.prompt_key[0:self.lmax_list[-1]], dim=-1)
        x_embed_norm = self.l2_normalize(cls_features, dim=-1)
        pred_task_id = task_id if self.training else torch.tensor(0)
        output['max_t'] = pred_task_id + 1
        last_class = self.lmax_list[-1]
        first_class = 0 if task_id == 0 or not self.training else self.lmax_list[-2]
        prompt_key = self.prompt_key[first_class:last_class]
        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)
        similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()).t()
        output['similarity'] = similarity
        similarity_top_k, indices = torch.topk(similarity, k=self.top_k_l, dim=1)
        raw = cls_features.unsqueeze(1)
        if self.training:
            key_norm = self.l2_normalize(self.prompt_key, dim=-1)
            current_key = key_norm[y]
            output['reduce_sim'] = torch.sum(current_key * x_embed_norm) / x_embed.shape[0]
            if task_id > 0:
                old_key_norm = self.l2_normalize(self.prompt_key[:self.lmax_list[-2]].clone(), dim=-1)
                old_similarity = torch.matmul(old_key_norm, x_embed_norm.t()).t()
                _, old_indices = torch.topk(old_similarity, k=1, dim=1)
                output['reduce_sim2'] = torch.sum(current_key * old_key_norm[old_indices[:, 0]]) / x_embed_norm.shape[0]
        raw = F.interpolate(raw, [self.embed_dim // self.num_heads])
        raw = raw.repeat(1, self.num_heads, 1).view(x_embed.shape[0], self.num_heads,
                                                   self.embed_dim // self.num_heads).unsqueeze(0)
        raw = torch.cat((raw, raw), dim=0)
        output['prompt_key_norm'], output['x_embed_norm'] = prompt_key_norm, x_embed_norm
        similarity_new = torch.zeros_like(similarity, device=similarity.device)
        for column in range(indices.shape[1]):
            # Preserve source batch-crossing advanced indexing; scatter would
            # change both prompts and gradients when examples choose other keys.
            similarity_new[:, indices[:, column]] = similarity_top_k[:, column]
        output['batched_prompt'] = self.compute_att_over_prompt(raw, 0, self.kmax_list[0],
                                                              layer_num, similarity_new, indices)
        output['cls_features'] = cls_features
        return output

    def _clamp_affine(self):
        # Preserve source .data replacement and clamp on inference as well.
        self.scaler.data = torch.clamp(self.scaler.data, min=.999, max=1.001)
        self.shifter.data = torch.clamp(self.shifter.data, min=-.0001, max=.0001)
        self.scaler_v.data = torch.clamp(self.scaler_v.data, min=.999, max=1.001)
        self.shifter_v.data = torch.clamp(self.shifter_v.data, min=-.0001, max=.0001)

    def compute_att_over_prompt(self, batched_prompt, start, stop, layer_num, similarity, indices):
        key_prompt = batched_prompt[0].permute(1, 0, 2)
        value_prompt = batched_prompt[1].permute(1, 0, 2)
        heads, batch_size, head_dim = key_prompt.shape
        new_key = torch.zeros((heads, batch_size, head_dim), device=key_prompt.device)
        new_value = torch.zeros((heads, batch_size, head_dim), device=value_prompt.device)
        for head in range(self.num_heads):
            for index in range(start, stop):
                new_key[head] += self.k_comp_gen[str(layer_num)][str(head)][index](key_prompt[head].unsqueeze(1)).squeeze(1) * similarity[:, index - start].unsqueeze(1)
                new_value[head] += self.v_comp_gen[str(layer_num)][str(head)][index](value_prompt[head].unsqueeze(1)).squeeze(1) * similarity[:, index - start].unsqueeze(1)
        self._clamp_affine()
        new_key, new_value = new_key.unsqueeze(-1), new_value.unsqueeze(-1)
        for length in range(1, self.length):
            scaled_key = (new_key[:, :, :, 0] * self.scaler[indices, length] + self.shifter[indices, length]).unsqueeze(-1)
            scaled_value = (new_value[:, :, :, 0] * self.scaler_v[indices, length] + self.shifter_v[indices, length]).unsqueeze(-1)
            new_key, new_value = torch.cat((new_key, scaled_key), dim=-1), torch.cat((new_value, scaled_value), dim=-1)
        result = torch.stack((new_key, new_value), dim=0).permute(2, 0, 4, 1, 3)
        self._clamp_affine()
        return result

    def get_extra_state(self):
        return {'version': 1, 'old_num_k': self.old_num_k, 'new_num_k': self.new_num_k,
                'old_num_c': self.old_num_c, 'new_num_c': self.new_num_c,
                'kmax_list': list(self.kmax_list), 'lmax_list': list(self.lmax_list)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.get_extra_state()) or state.get('version') != 1:
            raise ValueError('Invalid PROL prompt checkpoint state')
        count = len(state['lmax_list']) if isinstance(state['lmax_list'], list) else -1
        expected = {'version': 1, 'old_num_k': max(0, count - 1), 'new_num_k': count,
                    'old_num_c': max(0, count - 1) * self.numclasses_per_task,
                    'new_num_c': count * self.numclasses_per_task,
                    'kmax_list': [1] if count else [],
                    'lmax_list': [self.numclasses_per_task * index for index in range(1, count + 1)]}
        if not 0 <= count <= self.n_tasks or state != expected:
            raise ValueError('Inconsistent PROL prompt checkpoint lifecycle')
        for name in ('old_num_k', 'new_num_k', 'old_num_c', 'new_num_c', 'kmax_list', 'lmax_list'):
            setattr(self, name, state[name])
        for generators in (self.k_comp_gen, self.v_comp_gen):
            generators.requires_grad_(count < 2)
