"""Per-family block patches that reduce tokens between attention and MLP.

A patch swaps the class of each block for a thin subclass. Module names,
parameters and state dictionary keys stay untouched, and at degree 0 before
any reduction the original ``forward`` runs unchanged.
"""

from contextlib import contextmanager

from torch.nn import functional as F

from backbone import vit as mammoth_vit

from .controller import CompressionController
from .reducers import OPERATORS, Signals, planned


class _Compressed:
    """Bookkeeping shared by the patched blocks of every family."""

    def _enter(self, tokens):
        controller = self._tc
        if self._tc_index == 0:
            controller.begin_pass()
        controller.trace.append(tokens)
        return controller, planned(controller.current, tokens)

    @staticmethod
    def _bias(size, like):
        # Proportional attention: a merged token counts as its number of sources.
        return None if size is None else size.log()[:, None, None, :, 0].to(like.dtype)


class MammothBlock(_Compressed, mammoth_vit.Block):
    """``backbone.vit`` block: fused attention, no prompts."""

    def forward(self, x, **kwargs):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, **kwargs)
        if kwargs:
            raise NotImplementedError('LoRA-style block arguments are not supported with token compression')
        return self._reduced(x, controller, count)

    def _reduced(self, x, controller, count):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y).reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bias = self._bias(controller.size, q)
        # Same call as backbone.vit.Attention, plus the size bias.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attn.scale, dropout_p=attn.attn_drop.p)
        y = attn.proj_drop(attn.proj(y.transpose(1, 2).reshape(batch, tokens, width)))
        x = x + self.drop_path1(self.ls1(y))
        if count:
            signals = Signals(q, k, attn.scale, bias, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


def _mammoth():
    return None, mammoth_vit.VisionTransformer, mammoth_vit.Block, mammoth_vit.Attention, MammothBlock


def _coda():
    from .coda import FAMILY
    return FAMILY


def _sd_lora():
    from .sd_lora import FAMILY
    return FAMILY


# name -> loader of (encoder attribute on the network or None, encoder class, block class, attention class,
# patch). Method families are imported on demand, so the plain ViT works without the models package.
FAMILIES = {'mammoth-vit': _mammoth, 'coda-vit': _coda, 'sd-lora-vit': _sd_lora}


def find_encoder(net):
    """Return ``(family, encoder)`` for a supported network, by exact class."""
    for name, family in FAMILIES.items():
        attribute, encoder_class, *_ = family()
        encoder = net if attribute is None else getattr(net, attribute, None)
        if type(encoder) is encoder_class:
            return name, encoder
    raise TypeError(f'Token compression does not support {type(net).__name__}')


def attach(net, operator='tome', degree=0, bank=None):
    """Patch the encoder blocks of ``net`` and return their controller.

    ``bank`` is a FrozenToCom trained for this operator and these pretrained
    weights; the caller is responsible for checking its provenance.
    """
    if operator not in OPERATORS:
        raise ValueError(f'Unknown token reduction operator: {operator}')
    name, encoder = find_encoder(net)
    _, _, block_class, attention_class, patch = FAMILIES[name]()
    if getattr(encoder, 'token_compression', None) is not None:
        raise RuntimeError('Token compression is already attached to this encoder')
    for block in encoder.blocks:
        # An unknown subclass may bypass or reorder the patched forward.
        if type(block) is not block_class or type(block.attn) is not attention_class:
            raise TypeError(f'Unsupported block in {name}: {type(block).__name__}/{type(block.attn).__name__}')
    if bank is not None and len(bank.layers) != len(encoder.blocks):
        raise ValueError('ToCom and backbone depths differ')
    controller = CompressionController(OPERATORS[operator](), len(encoder.blocks), degree)
    controller.bank = bank
    for index, block in enumerate(encoder.blocks):
        block.__class__ = patch
        block._tc, block._tc_index = controller, index
    encoder.token_compression = controller
    return controller


@contextmanager
def compensate(net, source, target, scale):
    """Run ``net`` at ``target`` with the bank correcting a model trained at ``source``."""
    _, encoder = find_encoder(net)
    controller = encoder.token_compression
    if controller.bank is None:
        raise RuntimeError('No ToCom bank is attached to this encoder')
    # The bank is deliberately not a submodule, so it does not follow net.to().
    controller.bank.to(encoder.blocks[0].attn.qkv.weight.device)
    with controller.degree(target), controller.bank.applied(encoder, source, target, scale):
        yield controller


def detach(net):
    """Restore the original block classes."""
    name, encoder = find_encoder(net)
    block_class = FAMILIES[name]()[2]
    for block in encoder.blocks:
        block.__class__ = block_class
        del block._tc, block._tc_index
    del encoder.token_compression
