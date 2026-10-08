"""Load shared PROL Base weights on CPU; no image training or resource benchmark."""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/prol_prompt/cpu_weights.json')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MAMMOTH_DEVICE'] = 'cpu'
    sys.path.insert(0, str(ROOT))
    from scripts.cl_runtime import configure_native_environment
    configure_native_environment()
    from scripts.prol_protocol import file_hash, load_preparation_plan, verify_reference_source
    plan = load_preparation_plan()
    verify_reference_source()
    checkpoint = plan['common']['checkpoint']
    if file_hash(checkpoint['file']) != checkpoint['sha256']:
        raise ValueError('Common local NPZ hash mismatch')

    import torch
    from backbone.prol_vit import prol_vit
    # Complete Mammoth model-discovery imports before setting the experiment
    # seed: unrelated modules may initialize cached CLIP components on import.
    from models.prol_prompt_utils.network import ProlNetwork  # noqa: F401
    from scripts.ease_size_protocol import canonical_pretrained_digest, canonical_pretrained_state
    torch.set_num_threads(2)
    torch.manual_seed(1993)
    network = prol_vit(num_classes=100, prol_checkpoint=checkpoint['file'])
    native_rng = torch.get_rng_state().clone()
    branches = {'query': network.query, 'prompted': network.prompted}
    hashes = {name: canonical_pretrained_digest(branch) for name, branch in branches.items()}
    if any(value != checkpoint['canonical_pretrained_sha256'] for value in hashes.values()):
        raise ValueError('PROL initial encoder weights differ from the common canonical pin')
    counts = {name: sum(parameter.numel() for parameter in network.parameters()
                        if parameter.requires_grad == trainable)
              for name, trainable in (('trainable', True), ('frozen', False))}
    allowed = [name for name, parameter in network.prompted.named_parameters()
               if parameter.requires_grad and not name.startswith(('e_prompt.', 'head.'))]
    if not allowed or any(not (name.startswith('norm.') or '.norm1.' in name or '.norm2.' in name) for name in allowed):
        raise ValueError('PROL encoder trainability differs from source LayerNorm policy')
    if any(parameter.requires_grad for parameter in network.query.parameters()):
        raise ValueError('PROL query branch must be completely frozen')
    # Independent source constructor and original NPZ loader, not native math.
    from scripts.prol_reference_utils import load_author_code, build_author_branches
    code = load_author_code(plan['reference_source'])
    torch.manual_seed(1993)
    author = build_author_branches(code, checkpoint=checkpoint['file'])
    author_hashes = {name: canonical_pretrained_digest(branch) for name, branch in author.items()}
    if any(value != checkpoint['canonical_pretrained_sha256'] for value in author_hashes.values()):
        raise ValueError('Independent source encoder differs from common checkpoint')
    if not torch.equal(native_rng, torch.get_rng_state()):
        raise ValueError('Native/source Base initialization RNG differs')
    for name, native_branch in (('query', network.query), ('prompt', network.prompted)):
        native_state = {key: value for key, value in native_branch.state_dict().items() if isinstance(value, torch.Tensor)}
        source_state = author[name].state_dict()
        if native_state.keys() != source_state.keys() or any(not torch.equal(value, source_state[key])
                                                            for key, value in native_state.items()):
            raise ValueError('Complete native/source Base initial tensor states differ')
        if ({key: value.requires_grad for key, value in native_branch.named_parameters()}
                != {key: value.requires_grad for key, value in author[name].named_parameters()}):
            raise ValueError('Native/source Base initial trainability differs')
    record = {'status': 'passed', 'scope': 'CPU initial Base weight loading and structure only',
              'source_commit': plan['reference_commit'], 'checkpoint': checkpoint,
              'canonical_encoder_sha256': hashes,
              'independent_author_canonical_encoder_sha256': author_hashes,
              'complete_native_author_initial_tensors_bitexact': True,
              'native_author_initial_trainability_exact': True,
              'native_author_initial_cpu_rng_bitexact': True,
              'canonical_tensors_per_encoder': len(canonical_pretrained_state(network.query)),
              'parameters_before_task1': counts, 'allowed_trainable_encoder_norms': allowed,
              'initial_generator_parameters': sum(p.numel() for p in network.prompted.e_prompt.k_comp_gen.parameters())
                                              + sum(p.numel() for p in network.prompted.e_prompt.v_comp_gen.parameters()),
              'runtime_efficiency_measurement': False, 'cifar_training_executed': False,
              'real_image_inference_executed': False, 'gpu_computation_executed': False,
              'downloads_executed': False, 'credentials_read': False,
              'wandb_messages_sent': False, 'slack_messages_sent': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'parameters_before_task1': counts,
                      'canonical_encoder_sha256': hashes,
                      'initial_generator_parameters': record['initial_generator_parameters']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
