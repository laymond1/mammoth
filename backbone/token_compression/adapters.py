"""Per-family block patches that reduce tokens between attention and MLP.

A patch swaps the class of each block for a thin subclass. Module names,
parameters and state dictionary keys stay untouched, and at degree 0 before
any reduction the original ``forward`` runs unchanged. Encoders that drive
their blocks piecewise, without calling them, get one more swap on the module
that owns that loop.
"""

from collections import namedtuple
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


# attribute: dotted path of the encoder on the network (None: the network itself). host: None, or
# (dotted path or None for the encoder, class, patch) of the module that loops over the blocks itself.
# companions: further encoders of the network that follow the same controller and bank, each as
# (dotted path, encoder class, block class, attention class, patch).
Family = namedtuple('Family', 'attribute encoder block attention patch host companions', defaults=(None, ()))


def _mammoth():
    return Family(None, mammoth_vit.VisionTransformer, mammoth_vit.Block, mammoth_vit.Attention, MammothBlock)


def _load(module, name='FAMILY'):
    def family():
        from importlib import import_module
        return Family(*getattr(import_module(f'.{module}', __package__), name))
    return family


# Method families are imported on demand, so the plain ViT works without the models package.
FAMILIES = {'mammoth-vit': _mammoth, 'coda-vit': _load('coda'), 'sd-lora-vit': _load('sd_lora'),
            'inf-lora-vit': _load('low_rank', 'INF_LORA'), 'ewc-lora-vit': _load('low_rank', 'EWC_LORA'),
            'cps-vit': _load('cps'), 'rep-vit': _load('rep'), 'ease-vit': _load('ease')}


def _resolve(net, path):
    for name in path.split('.') if path else ():
        net = getattr(net, name, None)
    return net


def _host(net, encoder, family):
    return encoder if family.host[0] is None else _resolve(net, family.host[0])


def _companions(net, family):
    found = []
    for attribute, encoder_class, block_class, attention_class, patch in family.companions:
        encoder = _resolve(net, attribute)
        if type(encoder) is not encoder_class:
            raise TypeError(f'Token compression needs {attribute} to be a {encoder_class.__name__}')
        found.append((encoder, block_class, attention_class, patch))
    return found


def find_encoder(net):
    """Return ``(family, encoder)`` for a supported network, by exact class."""
    for name, load in FAMILIES.items():
        family = load()
        encoder = _resolve(net, family.attribute)
        if family.host is None:
            found = type(encoder) is family.encoder
        else:
            # The host carries its patched class while compression is attached.
            found = type(_host(net, encoder, family)) in family.host[1:] and (
                family.host[0] is None or type(encoder) is family.encoder)
        if found:
            return name, encoder
    raise TypeError(f'Token compression does not support {type(net).__name__}')


def encoders(net):
    """Return ``(family, [encoder, ...])``: the main encoder first, then those sharing its controller."""
    name, encoder = find_encoder(net)
    return name, [encoder, *(member for member, *_ in _companions(net, FAMILIES[name]()))]


def attach(net, operator='tome', degree=0, bank=None):
    """Patch the encoder blocks of ``net`` and return their controller.

    ``bank`` is a FrozenToCom trained for this operator and these pretrained
    weights; the caller is responsible for checking its provenance.
    """
    if operator not in OPERATORS:
        raise ValueError(f'Unknown token reduction operator: {operator}')
    name, encoder = find_encoder(net)
    family = FAMILIES[name]()
    if getattr(encoder, 'token_compression', None) is not None:
        raise RuntimeError('Token compression is already attached to this encoder')
    members = [(encoder, family.block, family.attention, family.patch), *_companions(net, family)]
    for member, block_class, attention_class, _ in members:
        if len(member.blocks) != len(encoder.blocks):
            raise ValueError('Encoders sharing one controller must have the same depth')
        for block in member.blocks:
            # An unknown subclass may bypass or reorder the patched forward.
            if type(block) is not block_class or type(block.attn) is not attention_class:
                raise TypeError(f'Unsupported block in {name}: {type(block).__name__}/{type(block.attn).__name__}')
    if bank is not None and len(bank.layers) != len(encoder.blocks):
        raise ValueError('ToCom and backbone depths differ')
    controller = CompressionController(OPERATORS[operator](), len(encoder.blocks), degree)
    controller.bank = bank
    for member, _, _, patch in members:
        for index, block in enumerate(member.blocks):
            block.__class__ = patch
            block._tc, block._tc_index = controller, index
        member.token_compression = controller
    if family.host is not None:
        _host(net, encoder, family).__class__ = family.host[2]
    return controller


@contextmanager
def compensate(net, source, target, scale):
    """Run ``net`` at ``target`` with the bank correcting a model trained at ``source``.

    ``source`` is one degree, or one per encoder pass of a forward call in the
    order the passes run (the query pass, then the prompted pass), for models
    whose passes were trained at different degrees. Run one forward per context.
    """
    name, encoder = find_encoder(net)
    controller = encoder.token_compression
    if controller.bank is None:
        raise RuntimeError('No ToCom bank is attached to this encoder')
    sources = (source,) if isinstance(source, int) else tuple(source)
    if not sources:
        raise ValueError('At least one training degree is required')
    for value in (*sources, target):
        controller.bank.covered(value)
    members = [encoder, *(member for member, *_ in _companions(net, FAMILIES[name]()))]
    # The bank is deliberately not a submodule, so it does not follow net.to().
    controller.bank.to(encoder.blocks[0].attn.qkv.weight.device)
    controller.sources, controller.passes = sources, 0
    try:
        with controller.degree(target), controller.bank.applied(
                members, sources[0] if len(sources) == 1 else controller.source, target, scale):
            yield controller
        if len(sources) > 1 and controller.passes not in (0, len(sources)):
            raise RuntimeError(f'{len(sources)} training degrees were given, the model ran {controller.passes} passes')
    finally:
        controller.sources = None


def detach(net):
    """Restore the original block classes."""
    name, encoder = find_encoder(net)
    family = FAMILIES[name]()
    if family.host is not None:
        _host(net, encoder, family).__class__ = family.host[1]
    for member, block_class, _, _ in [(encoder, family.block, None, None), *_companions(net, family)]:
        for block in member.blocks:
            block.__class__ = block_class
            del block._tc, block._tc_index
        del member.token_compression
