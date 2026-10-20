"""ViT-Ti/S/B with every token-reduction path of the study.

Parameter names follow timm, so the canonical AugReg weights load as they are. The dense
math is that of ``backbone/vit.py`` (scaled-dot-product attention, LayerNorm eps 1e-6,
exact GELU). In-block reduction runs between attention and MLP with the size-weighted
attention bias, exactly as ``MammothBlock._reduced`` of the tocom-plugin patch.

Layers are numbered from 1 as in the token-ranking study: the score "at layer L" is read
from the attention inside block L, so it needs blocks 1..L-1 and the query/key/value of
block L.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .reducers import OPERATORS, Signals, planned

SIZES = {'tiny': (192, 3), 'small': (384, 6), 'base': (768, 12)}
PATCHES, DEPTH = 196, 12


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads, self.scale = heads, (width // heads) ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def heads(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def mix(self, q, k, v, bias=None):
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        batch, heads, tokens, dim = y.shape
        return self.proj(y.transpose(1, 2).reshape(batch, tokens, heads * dim))


class Mlp(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(width, eps=1e-6), Attention(width, heads)
        self.norm2, self.mlp = nn.LayerNorm(width, eps=1e-6), Mlp(width)

    def forward(self, x):
        x = x + self.attn.mix(*self.attn.heads(self.norm1(x)))
        return x + self.mlp(self.norm2(x))

    def reduced(self, x, state, index):
        """One block of an in-block operator; ``state`` carries sizes and the token trace of the pass.

        Operators reduce between attention and MLP, as ToMe does, unless they set ``after_mlp``
        (Representation Shift scores tokens by their MLP update and prunes the block output).
        """
        state.trace.append(x.shape[1])
        count = planned(state.request(index, x.shape[1]), x.shape[1])
        if count == 0 and state.size is None:
            return self(x)
        q, k, v = self.attn.heads(self.norm1(x))
        # Proportional attention: a merged token counts as its number of sources.
        bias = None if state.size is None else state.size.log()[:, None, None, :, 0].to(q.dtype)
        x = x + self.attn.mix(q, k, v, bias)
        after = getattr(state.operator, 'after_mlp', False)
        if count and not after:
            signals = BlockSignals(q, k, v, self.attn.scale, bias, block=index, depth=state.depth)
            x, state.size = state.operator(x, state.size, count, signals)
        update = self.mlp(self.norm2(x))
        if count and after:
            return state.operator.after(x, update, state, count)
        return x + update


class BlockSignals(Signals):
    """The plug-in's signals plus the value tensor, for operators that weight attention by value norms."""

    def __init__(self, q, k, v, scale, bias=None, prefix=0, block=0, depth=12):
        super().__init__(q, k, scale, bias, prefix, block, depth)
        self.v = v

    def significance(self):
        """ATS/PPT significance of every patch: head-summed CLS attention (with the size bias) times the value norm."""
        batch, heads, tokens, dim = self.v.shape
        attention = self.cls_attention() * heads  # cls_attention averages over heads
        norm = torch.linalg.vector_norm(self.v.transpose(1, 2).reshape(batch, tokens, heads * dim), dim=2)[:, self.prefix + 1:]
        return attention * norm


class Reduction:
    """Per-pass state of an in-block operator.

    ``degree`` removes that many tokens in every block; ``schedule`` gives the request per block instead
    (a count, or a fraction of the current tokens when below 1, as Representation Shift writes it).
    """

    def __init__(self, operator, degree=0, depth=DEPTH, schedule=None):
        if isinstance(operator, str):
            from .operators import ALL_OPERATORS
            operator = ALL_OPERATORS[operator]()
        self.operator, self.degree, self.depth, self.schedule = operator, degree, depth, schedule
        self.size, self.trace = None, []

    def request(self, index, tokens):
        if self.schedule is None:
            return self.degree
        value = self.schedule[index]
        return int(tokens * value) if 0 < value < 1 else int(value)


class PatchEmbed(nn.Module):
    """16x16 patches through the convolution's weights as one matrix product.

    Same arithmetic as the stride-16 convolution but run as a GEMM, so no convolution library
    is involved on any device (cuDNN in true float32 crashed this study's environment, see
    ``benchmarks/token_efficiency/NOTES.md``).
    """

    def __init__(self, width):
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=16, stride=16)

    def forward(self, images):
        batch, _, height, width = images.shape
        rows, cols = height // 16, width // 16
        patches = images.reshape(batch, 3, rows, 16, cols, 16).permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, 768)
        return F.linear(patches, self.proj.weight.reshape(self.proj.out_channels, 768), self.proj.bias)


def critical_score(attention, q, k, v):
    """CPS-Prompt's importance of every patch: head-summed CLS attention times the value norm, in float32."""
    batch, heads, tokens, dim = v.shape
    cls = ((q[:, :, :1] @ k.transpose(-2, -1)) * attention.scale).softmax(dim=-1)[:, :, 0].float().sum(dim=1)
    norm = torch.linalg.vector_norm(v.float().transpose(1, 2).reshape(batch, tokens, heads * dim), dim=2)
    return (cls * norm)[:, 1:]


def with_cls(indices):
    """Token indices of the kept patches, sorted, with the class token in front."""
    indices = indices.sort(dim=1).values + 1
    return torch.cat((indices.new_zeros(len(indices), 1), indices), dim=1)


def gather_tokens(x, index):
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class VisionTransformer(nn.Module):
    def __init__(self, size='tiny', classes=1000, depth=DEPTH):
        super().__init__()
        width, heads = SIZES[size]
        self.size, self.width = size, width
        self.patch_embed = PatchEmbed(width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, PATCHES + 1, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head = nn.Linear(width, classes)

    # ---- pieces ----
    def embed(self, images):
        x = self.patch_embed(images)
        return torch.cat((self.cls_token.expand(len(x), -1, -1), x), dim=1) + self.pos_embed

    def classify(self, x):
        return self.head(self.norm(x[:, 0]))

    def run(self, x, start=0, stop=None):
        for block in self.blocks[start:stop]:
            x = block(x)
        return x

    # ---- paths ----
    def forward(self, images, reduction=None):
        """Dense, or with an in-block operator when ``reduction`` is a :class:`Reduction`."""
        x = self.embed(images)
        if reduction is None:
            return self.classify(self.run(x))
        reduction.size, reduction.trace = None, []
        for index, block in enumerate(self.blocks):
            x = block.reduced(x, reduction, index)
        return self.classify(x)

    def scores(self, images, layer, embedded=None):
        """Critical score of every patch read at ``layer`` (1-indexed), and the state to continue from it."""
        x = self.run(self.embed(images) if embedded is None else embedded, 0, layer - 1)
        block = self.blocks[layer - 1]
        q, k, v = block.attn.heads(block.norm1(x))
        return critical_score(block.attn, q, k, v), (x, q, k, v)

    def forward_kept(self, images, kept, embedded=None):
        """Restart: run all blocks on the class token and the ``kept`` patch indices (batch, k), 0-based."""
        x = self.embed(images) if embedded is None else embedded
        return self.classify(self.run(gather_tokens(x, with_cls(kept))))

    def forward_continue(self, layer, state, kept):
        """Continue: finish the attention of block ``layer`` on all tokens, keep the chosen ones before its MLP."""
        x, q, k, v = state
        block = self.blocks[layer - 1]
        x = gather_tokens(x + block.attn.mix(q, k, v), with_cls(kept))
        x = x + block.mlp(block.norm2(x))
        return self.classify(self.run(x, layer))


def build(size='tiny', state=None, device='cpu', dtype=torch.float32):
    model = VisionTransformer(size)
    if state is not None:
        model.load_state_dict({name: torch.as_tensor(value) for name, value in state.items()}, strict=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
