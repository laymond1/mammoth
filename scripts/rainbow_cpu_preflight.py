"""Load pinned RainbowPrompt Base weights on CPU without forwarding real images."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/cpu_weights.json')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MAMMOTH_DEVICE'] = 'cpu'
    sys.path.insert(0, str(ROOT))
    from scripts.cl_runtime import configure_native_environment
    configure_native_environment()
    import torch
    from backbone.rainbow_vit import rainbow_vit
    from scripts.ease_size_protocol import canonical_pretrained_digest, canonical_pretrained_state

    torch.set_num_threads(2)
    torch.manual_seed(1993)
    plan = json.loads((ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/validation_plan.json').read_text())
    checkpoint = plan['common']['checkpoint']
    digest = hashlib.sha256()
    with Path(checkpoint['file']).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != checkpoint['sha256']:
        raise ValueError('Pinned local NPZ SHA-256 mismatch')
    network = rainbow_vit(num_classes=100, rainbow_checkpoint=checkpoint['file'])
    hashes = {name: canonical_pretrained_digest(branch) for name, branch in
              (('query', network.query), ('prompt', network.prompt))}
    if any(value != checkpoint['canonical_pretrained_sha256'] for value in hashes.values()):
        raise ValueError('RainbowPrompt encoder canonical weights differ from the common pretrained pin')
    counts = {name: sum(parameter.numel() for parameter in network.parameters()
                        if parameter.requires_grad == trainable)
              for name, trainable in (('trainable', True), ('frozen', False))}
    cache = network.prompt.rainbow_prompt.stored_rainbow_prompts
    record = {'status': 'passed', 'scope': 'CPU weight loading and structure only',
              'source_commit': plan['reference_commit'], 'checkpoint': checkpoint,
              'canonical_encoder_sha256': hashes,
              'canonical_tensors_per_encoder': len(canonical_pretrained_state(network.query)),
              'parameters': counts, 'cached_prompt_bytes': cache.numel() * cache.element_size(),
              'runtime_efficiency_measurement': False, 'cifar_training_executed': False,
              'real_image_inference_executed': False, 'gpu_computation_executed': False,
              'downloads_executed': False, 'wandb_messages_sent': False, 'slack_messages_sent': False,
              'protocol': network.get_extra_state()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'parameters': counts,
                      'canonical_encoder_sha256': hashes, 'cached_prompt_bytes': record['cached_prompt_bytes']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
