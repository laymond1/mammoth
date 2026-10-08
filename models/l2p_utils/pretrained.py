"""Pinned local encoder checks shared by L2P and DualPrompt.

L2P prepends prompt positions. Its canonical encoder position table therefore
contains the CLS position and the final 196 image-patch positions only.
"""
import hashlib

import torch


def checkpoint_spec(vit_size, checkpoint):
    from models.low_rank_utils.factory import checkpoint_spec as shared_spec
    return shared_spec(vit_size, checkpoint)


def canonical_encoder_state(model, prompt_positions=False):
    from models.low_rank_utils.pretrained import canonical_pretrained_state
    state = canonical_pretrained_state(model)
    if len(state) != 150 or len(model.blocks) != 12:
        raise ValueError('Prompt methods require the complete 12-block, 150-tensor encoder')
    if prompt_positions:
        positions = state['pos_embed']
        expected_tokens = 197 + getattr(model, 'total_prompt_len', 0)
        # Before the first image forward, L2P stores the token count in the
        # position-table geometry rather than total_prompt_len.
        if hasattr(model, 'prompt'):
            expected_tokens = 197 + model.prompt.length * model.prompt.top_k
        if positions.shape != (1, expected_tokens, model.embed_dim):
            raise ValueError('L2P prompt/image position geometry is invalid')
        state['pos_embed'] = torch.cat((positions[:, :1], positions[:, -196:]), dim=1)
    for name, tensor in state.items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f'Prompt pretrained encoder tensor is not finite FP32: {name}')
    return state


def canonical_encoder_digest(model, prompt_positions=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(canonical_encoder_state(model, prompt_positions).items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def verify_pretrained_branches(query, prompted, spec, *, l2p=False):
    """Require both encoders to equal the pinned NPZ's canonical tensors."""
    hashes = {'query': canonical_encoder_digest(query),
              'prompt': canonical_encoder_digest(prompted, prompt_positions=l2p)}
    if any(value != spec['canonical_sha256'] for value in hashes.values()):
        raise ValueError('Query or prompt encoder differs from the pinned pretrained tensor digest')
    return hashes
