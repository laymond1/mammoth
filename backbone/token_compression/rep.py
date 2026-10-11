"""REP-CODA encoder: REP's own schedule in training, the plug-in's per-block reduction on request.

REP merges inside its encoder loop, so the loop is patched. Without an
explicit degree the encoder runs exactly as REP does: progressive merging
(AToM) and layer dropping as configured, in training and in its own inference.
Inside a degree context the plug-in replaces AToM by a uniform reduction, so
that a REP model is inferred at the same degrees as every other method. The
query encoder is a separate module; it follows the same controller, which
leaves it unreduced unless a degree is requested.

A nonzero default degree is the other way to train: the plug-in reduces in
training too, and REP must then run with ``rep_atom=0``.
"""

import torch

from models.coda_prompt_utils import vit as coda_vit
from models.rep_prompt_utils import vit as rep_vit

from .coda import CodaBlock
from .reducers import planned


class RepEncoder(rep_vit.REPVisionTransformer):
    def forward(self, images, prompt, query, *, train=False, atom=True, ald=True, maximum=16,
                step=0, rng=None, minimum=.5, gamma=.001, alpha=.9, threshold=8):
        controller = self.token_compression
        controller.begin_pass()
        own = not controller.overridden and controller.default_degree == 0
        if atom and not own and not controller.overridden:
            raise RuntimeError('REP must run with rep_atom=0 when the plug-in reduces tokens in training')
        if own or controller.current == 0:
            # REP as it is; an explicit degree 0 means no reduction at all, so its merging is switched off.
            output = super().forward(images, prompt, query, train=train, atom=atom and own, ald=ald,
                                     maximum=maximum, step=step, rng=rng, minimum=minimum, gamma=gamma,
                                     alpha=alpha, threshold=threshold)
            controller.trace = list(output[2]['input_tokens'])
            return output
        x = self.patch_embed(images)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.shape[1]])
        draws = torch.rand(len(self.blocks), generator=rng.generator('cpu')) if train and ald else None
        loss = x.new_zeros(1)
        trace = {'step': step, 'input_tokens': [], 'output_tokens': [], 'merged_tokens': [],
                 'executed': [], 'keeping_probabilities': [], 'prefix_tokens': []}
        for index, block in enumerate(self.blocks):
            before = x.shape[1]
            count = planned(controller.current, before)
            probability = rep_vit.keeping_probability(step, count, minimum=minimum, gamma=gamma, alpha=alpha,
                                                      threshold=threshold) if train and ald else 1.
            execute = draws is None or draws[index].item() < probability
            prefix_length = 0
            if execute:
                prefix, penalty, x = prompt(query, index, x, train=train)
                loss = loss + penalty
                prefix_length = 0 if prefix is None else prefix[0].shape[1]
                x = block._reduced(x, controller, count, prefix)
            trace['input_tokens'].append(before)
            trace['output_tokens'].append(x.shape[1])
            trace['merged_tokens'].append(before - x.shape[1])
            trace['executed'].append(bool(execute))
            trace['keeping_probabilities'].append(probability)
            trace['prefix_tokens'].append(prefix_length)
        controller.trace = list(trace['input_tokens'])
        return self.norm(x), loss, trace


FAMILY = ('feat', rep_vit.REPVisionTransformer, coda_vit.Block, rep_vit.REPAttention, CodaBlock,
          (None, rep_vit.REPVisionTransformer, RepEncoder),
          (('query_encoder', coda_vit.VisionTransformer, coda_vit.Block, coda_vit.Attention, CodaBlock),))
