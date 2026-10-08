"""Active RainbowPrompt math from author commit 0b84d3e3738669fe4bb516ccc1c526a535576ad3.

The source's task conditioning, always-active evolution dropout, detached past
knowledge and pre-update prompt cache are preserved deliberately. This module
does not introduce a gate or task-boundary component freezing.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RainbowPrompt(nn.Module):
    def __init__(self, length=20, embed_dim=768, embedding_key='cls', prompt_init='uniform', prompt_pool=True,
                 prompt_key=True, pool_size=10, top_k=1, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=None, use_prefix_tune_for_e_prompt=True, num_heads=12, same_key_value=False,
                 cls_rank=None, prompt_rank=None, prompt_tune_idx=None, n_tasks=10, D1=56,
                 relation_type='attention', use_linear=True, KI_iter=10, self_attn_idx=None, D2=96):
        super().__init__()
        prompt_tune_idx = tuple(range(12)) if prompt_tune_idx is None else tuple(prompt_tune_idx)
        self_attn_idx = tuple(range(6)) if self_attn_idx is None else tuple(self_attn_idx)
        counts = (length, embed_dim, pool_size, top_k, n_tasks, D1, D2, KI_iter, num_heads)
        if (any(type(value) is not int or value < 1 for value in counts)
                or length % 2 or embed_dim % num_heads or top_k != 1 or pool_size < n_tasks
                or not prompt_tune_idx or prompt_tune_idx != tuple(range(len(prompt_tune_idx)))
                or any(type(index) is not int for index in prompt_tune_idx)
                or any(type(index) is not int or index not in prompt_tune_idx for index in self_attn_idx)
                or len(set(self_attn_idx)) != len(self_attn_idx)
                or num_layers is not None and (type(num_layers) is not int or num_layers != len(prompt_tune_idx))):
            raise ValueError('RainbowPrompt requires an even prefix, top_k=1 and contiguous layers with valid task dimensions')
        if (embedding_key != 'cls' or prompt_init != 'uniform' or prompt_key_init != 'uniform'
                or prompt_pool is not True or prompt_key is not True or use_prefix_tune_for_e_prompt is not True
                or use_linear is not True or relation_type != 'attention' or batchwise_prompt is not False
                or same_key_value is not False or cls_rank is not None or prompt_rank is not None):
            raise ValueError('Only the active RainbowPrompt CLS-key, linear attention, independent-prefix recipe is supported')
        self.length, self.embed_dim = length, embed_dim
        self.prompt_pool, self.prompt_key = prompt_pool, prompt_key
        self.embedding_key, self.prompt_init = embedding_key, prompt_init
        self.pool_size, self.top_k = pool_size, top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers, self.num_heads = len(prompt_tune_idx), num_heads
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.same_key_value = same_key_value
        self.prompt_tune_idx, self.n_tasks = prompt_tune_idx, n_tasks
        self.D1, self.D2 = D1, D2
        self.relation_type, self.use_linear = relation_type, use_linear
        self.KI_iter, self.self_attn_idx = KI_iter, self_attn_idx
        self.register_buffer('stored_rainbow_prompts', torch.zeros(n_tasks, len(prompt_tune_idx), length, embed_dim))
        self.query_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in prompt_tune_idx])
        self.key_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in prompt_tune_idx])
        self.value_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in prompt_tune_idx])
        self.dense = nn.ModuleList([nn.Linear(D2, embed_dim) for _ in prompt_tune_idx])
        self.fc1 = nn.ModuleList([nn.Linear(embed_dim, D1) for _ in prompt_tune_idx])
        self.fc2 = nn.ModuleList([nn.Linear(D1, embed_dim) for _ in prompt_tune_idx])
        for layer in prompt_tune_idx:
            setattr(self, f'base_knowledge_{layer}', self.tensor_matrix(pool_size, length, embed_dim))
        self.base_key = self.tensor_matrix(n_tasks, embed_dim, None)

    def tensor_matrix(self, a, b, c):
        parameter = nn.Parameter(torch.empty((a, b) if c is None else (a, b, c)))
        nn.init.uniform_(parameter)
        return parameter

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        inverse = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * inverse

    def freeze_components(self, task_id):
        """Source compatibility helper; the active training flow does not call it."""
        if task_id > 0:
            for collection in (self.query_matcher, self.key_matcher, self.value_matcher, self.dense):
                for layer in collection:
                    for parameter in layer.parameters():
                        parameter.requires_grad = False

    def task_conditioning_step(self, base_knowledge, current_task_embed):
        # Each token receives the same task key. Preserve the author's expanded
        # key matmul/softmax, including the resulting uniform token averaging.
        key = current_task_embed.unsqueeze(0).unsqueeze(0)
        key = key.expand(base_knowledge.size(0), base_knowledge.size(1), -1)
        relevance = F.softmax(torch.matmul(base_knowledge, key.transpose(-1, -2)), dim=-1)
        return torch.matmul(relevance, base_knowledge)

    def Prompt_Evolution(self, layer, attended_prev, attended_curr, d_model, d_ff, dropout=.1):
        def transform(query, key, value):
            query = self.query_matcher[layer](query)
            key = self.key_matcher[layer](key)
            value = self.value_matcher[layer](value)
            scale = torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32).to(query.device))
            weights = F.softmax(torch.matmul(query, key.transpose(1, 2)) / scale, dim=-1)
            output = torch.matmul(weights, value)
            query_t, key_t = query.transpose(1, 2), key.transpose(1, 2)
            scale_t = torch.sqrt(torch.tensor(query_t.shape[-1], dtype=torch.float32).to(query.device))
            weights_t = F.softmax(torch.matmul(query_t, key_t.transpose(1, 2)) / scale_t, dim=-1)
            output = torch.matmul(weights_t, output.transpose(1, 2)).transpose(1, 2)
            return self.dense[layer](output)

        def evolve(previous, current):
            attention = F.dropout(transform(current, previous, previous), dropout, training=True)
            first = F.layer_norm(previous + attention, [d_model])
            feedforward = self.fc2[layer](F.relu(self.fc1[layer](first)))
            feedforward = F.dropout(feedforward, dropout, training=True)
            return F.layer_norm(first + feedforward, [d_model])

        if layer not in self.self_attn_idx:
            results = []
            for task in range(self.task_id + 1):
                previous = attended_curr if task == self.task_id else attended_prev[task * self.top_k:task * self.top_k + self.top_k]
                results.append(evolve(previous, attended_curr))
            return torch.cat(results, dim=0)
        previous = attended_curr
        for _ in range(self.KI_iter):
            previous = evolve(previous, attended_curr)
        return previous

    def forward(self, x_embed, layer, previous_mask=None, cls_features=None, task_id=None,
                cur_id=None, train=False, p_type=None):
        if (type(layer) is not int or layer not in self.prompt_tune_idx or type(cur_id) is not int
                or not 0 <= cur_id < self.n_tasks or cls_features is None or cls_features.ndim != 2
                or cls_features.shape[1] != self.embed_dim or cls_features.shape[0] < 1
                or type(train) is not bool or p_type not in (None, 'Unique', 'Rainbow')):
            raise ValueError('RainbowPrompt needs a valid learned-task range, layer and batch CLS features')
        knowledge, keys = getattr(self, f'base_knowledge_{layer}'), self.base_key
        embedding = self.l2_normalize(cls_features, dim=-1)
        result = {}
        if p_type == 'Rainbow' and not train:
            scores = []
            for task in range(cur_id + 1):
                key = self.l2_normalize(keys[task], dim=-1)
                scores.append(torch.sum(torch.matmul(key, embedding.t())) / embedding.shape[0])
            self.task_id = int(torch.argmax(torch.stack(scores)))
            prompt = self.stored_rainbow_prompts[self.task_id, layer]
        else:
            self.task_id = cur_id
            current = knowledge[cur_id * self.top_k:cur_id * self.top_k + self.top_k]
            key = self.l2_normalize(keys[cur_id], dim=-1)
            result['sim_loss'] = torch.sum(torch.matmul(key, embedding.t())) / embedding.shape[0]
            if p_type == 'Rainbow':
                previous = current if cur_id == 0 else knowledge[:cur_id * self.top_k].detach().clone()
                evolved = self.Prompt_Evolution(layer, self.task_conditioning_step(previous, key),
                    self.task_conditioning_step(current, key), self.embed_dim, self.D1)
                prompt = torch.mean(evolved, dim=0)
                with torch.no_grad():
                    self.stored_rainbow_prompts[cur_id, layer].copy_(prompt)
            else:
                prompt = torch.mean(current, dim=0)
        prompt = prompt.expand(embedding.shape[0], -1, -1)
        result['batched_prompt'] = [prompt[:, :self.length // 2, :], prompt[:, self.length // 2:, :]]
        return result
