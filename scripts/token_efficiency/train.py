"""Cost of one training step (forward, backward, optimizer update) under token reduction.

Three ways continual-learning methods train a ViT, as compute patterns rather than full methods:

  full    every weight trains (ER, LwF-style fine-tuning)
  lora    frozen weights, rank-10 LoRA on every qkv projection (InfLoRA-style)
  prompt  frozen weights; a gradient-free query pass gives the class-token query, and prefix
          prompts (length 8: 4 keys, 4 values) enter blocks 1-5 (CODA-Prompt-style); with
          ``kept`` + ``cps`` at layer 12 the selection reads the query pass, as CPS-Prompt does

Each uses a fresh 100-way head and AdamW; labels are random. Reduction specs are those of
``methods.py``; selections run without gradient, the trained pass on the reduced tokens.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .reducers import Signals, planned
from .vit import PATCHES, Reduction, critical_score, gather_tokens, with_cls

CLASSES, RANK, PROMPT_LAYERS, PROMPT_LENGTH, POOL = 100, 10, 5, 8, 100


class LoRALinear(nn.Module):
    def __init__(self, base, rank=RANK):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(rank, base.in_features).normal_(std=1 / math.sqrt(base.in_features)))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.down), self.up)

    @property
    def weight(self):
        return self.base.weight


class CodaPrompts(nn.Module):
    """Prompt components mixed by cosine attention of the query (CODA-Prompt's prompt formation)."""

    def __init__(self, width, layers=PROMPT_LAYERS, length=PROMPT_LENGTH, pool=POOL):
        super().__init__()
        self.components = nn.Parameter(torch.randn(layers, pool, length, width) * 0.02)
        self.keys = nn.Parameter(torch.randn(layers, pool, width) * 0.02)
        self.attention = nn.Parameter(torch.randn(layers, pool, width) * 0.02)

    def forward(self, query):
        # query: (batch, width) -> per layer prompts (layers, batch, length, width)
        weights = F.cosine_similarity(query[:, None, None, :] * self.attention[None], self.keys[None], dim=-1)
        return torch.einsum('blk,lkpd->lbpd', weights, self.components)


def attention_with_prefix(block, x, prefix, bias):
    """Block attention where ``prefix`` (batch, length, width) adds keys and values; returns (x, q, k, v)."""
    q, k, v = block.attn.heads(block.norm1(x))
    if prefix is not None:
        batch, length, width = prefix.shape
        heads = block.attn.num_heads
        half = length // 2
        pk = prefix[:, :half].reshape(batch, half, heads, width // heads).transpose(1, 2)
        pv = prefix[:, half:].reshape(batch, length - half, heads, width // heads).transpose(1, 2)
        keys, values = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
        if bias is not None:
            bias = torch.cat((bias.new_zeros(*bias.shape[:-1], half), bias), dim=-1)
    else:
        keys, values = k, v
    return x + block.attn.mix(q, keys, values, bias), q, k, v


def run_blocks(model, x, prompts=None, reduction=None, start=0, stop=None):
    """Blocks ``start``..``stop`` with optional prompts and an optional in-block operator."""
    stop = len(model.blocks) if stop is None else stop
    for index in range(start, stop):
        block = model.blocks[index]
        prefix = prompts[index] if prompts is not None and index < len(prompts) else None
        if reduction is None:
            if prefix is None:
                x = block(x)
            else:
                x, _, _, _ = attention_with_prefix(block, x, prefix, None)
                x = x + block.mlp(block.norm2(x))
            continue
        reduction.trace.append(x.shape[1])
        count = planned(reduction.degree, x.shape[1])
        bias = None if reduction.size is None else reduction.size.log()[:, None, None, :, 0].to(x.dtype)
        x, q, k, _ = attention_with_prefix(block, x, prefix, bias)
        if count:
            signals = Signals(q, k, block.attn.scale, bias, block=index, depth=reduction.depth)
            x, reduction.size = reduction.operator(x, reduction.size, count, signals)
        x = x + block.mlp(block.norm2(x))
    return x


class Trainer:
    """Builds the trainable parts for ``kind`` on ``model`` and returns a step function for ``spec``."""

    def __init__(self, model, kind, spec):
        self.model, self.kind, self.spec = model, kind, spec
        width = model.width
        model.head = nn.Linear(width, CLASSES).to(model.pos_embed.device, model.pos_embed.dtype)
        if kind == 'full':
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
            model.head.requires_grad_(True)
        self.prompts = None
        if kind == 'lora':
            for block in model.blocks:
                block.attn.qkv = LoRALinear(block.attn.qkv).to(model.pos_embed.device, model.pos_embed.dtype)
        elif kind == 'prompt':
            self.prompts = CodaPrompts(width).to(model.pos_embed.device, model.pos_embed.dtype)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        if self.prompts is not None:
            params += list(self.prompts.parameters())
        self.trainable = sum(p.numel() for p in params)
        self.optimizer = torch.optim.AdamW(params, lr=1e-4)

    def query(self, x):
        """Gradient-free dense query pass of the prompt methods: class-token feature and layer-12 scores."""
        with torch.no_grad():
            tokens = self.model.run(self.model.embed(x), 0, 11)
            block = self.model.blocks[11]
            q, k, v = block.attn.heads(block.norm1(tokens))
            score = critical_score(block.attn, q, k, v)
            tokens = tokens + block.attn.mix(q, k, v)
            tokens = tokens + block.mlp(block.norm2(tokens))
            return self.model.norm(tokens[:, 0]), score

    def logits(self, x):
        model, spec = self.model, self.spec
        kind = spec['kind']
        prompts, score12 = None, None
        if self.prompts is not None:
            query, score12 = self.query(x)
            prompts = self.prompts(query)
        if kind in ('dense', 'inblock'):
            reduction = Reduction(spec['operator'], spec['degree']) if kind == 'inblock' else None
            return model.classify(run_blocks(model, model.embed(x), prompts, reduction))
        if kind == 'kept':
            embedded = model.embed(x)
            with torch.no_grad():
                if spec['selector'] == 'cps' and spec['layer'] == 12 and score12 is not None:
                    score = score12
                elif spec['selector'] == 'cps':
                    score = model.scores(x, spec['layer'], embedded.detach())[0]
                elif spec['selector'] == 'random':
                    score = torch.rand(len(x), PATCHES, device=x.device)
                else:
                    raise ValueError(f"Training selector {spec['selector']} is not set up")
                kept = with_cls(score.topk(spec['keep'], dim=1).indices)
            return model.classify(run_blocks(model, gather_tokens(embedded, kept), prompts))
        if kind == 'continue':
            layer = spec['layer']
            x = run_blocks(model, model.embed(x), prompts, None, 0, layer - 1)
            block = model.blocks[layer - 1]
            prefix = prompts[layer - 1] if prompts is not None and layer - 1 < len(prompts) else None
            x, q, k, v = attention_with_prefix(block, x, prefix, None)
            with torch.no_grad():
                kept = with_cls(critical_score(block.attn, q, k, v).topk(spec['keep'], dim=1).indices)
            x = gather_tokens(x, kept)
            x = x + block.mlp(block.norm2(x))
            return model.classify(run_blocks(model, x, prompts, None, layer))
        raise ValueError(f'Training does not support {kind}')

    def step(self, x, labels, scaler=None, autocast=None):
        self.optimizer.zero_grad(set_to_none=True)
        if autocast is not None:
            with autocast:
                loss = F.cross_entropy(self.logits(x), labels)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss = F.cross_entropy(self.logits(x), labels)
            loss.backward()
            self.optimizer.step()
        return loss.detach()
