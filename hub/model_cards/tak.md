---
language: en
license: mit
library_name: pytorch
tags:
- continual-learning
- task-arithmetic
- kfac
- clip
- mammoth
pipeline_tag: image-classification
---

# TAK

This repository hosts artifacts for **TAK** in Mammoth (`--model tak`).

TAK v2 applies Task Arithmetic in a continual-learning setup and regularizes task-vector interactions with a **dataless** approximation based on **Kronecker-Factored Approximate Curvature (KFAC)** to reduce representation drift and interference.

## Paper

- **Title**: Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature
- **Venue**: ICLR 2026
- **arXiv**: https://arxiv.org/abs/2602.17385

## What is stored here

This repository is intended to store artifacts needed to reproduce or run TAK v2, such as:

- Fisher/KFAC cache files,
- task vectors,
- classifier heads and metadata,
- optional checkpoints and run notes.

For Fisher loading via Mammoth, keep naming consistent with the loader expectations, e.g.:

- `<dataset>_task_<task_id>_aaT.pt`
- `<dataset>_task_<task_id>_ggT.pt`
- `<dataset>_task_<task_id>_ffT.pt`
- `<dataset>_task_<task_id>_num_aaT.pt`
- `<dataset>_task_<task_id>_num_ggT.pt`

## How to use with Mammoth

Example command with Fisher cache hosted on this repo:

```bash
uv run python main.py \
  --model tak \
  --dataset=seq-8visio \
  --load_fisher 1 \
  --fisher_cache hf://aimagelab-ta/TAK/vitb16/fisher_8vision/kfac/mc_full@main \
  --alpha_merging 8.0 \
  --batch_size 32 --virtual_bs_n 4
```

If you need to upload artifacts from local storage:

```bash
uv run python scripts/upload_to_hf.py \
  --repo-id aimagelab-ta/TAK \
  --repo-type model \
  --local-dir /path/to/local/fisher \
  --remote-dir fisher \
  --pattern "**/*"
```

## Method overview

- Continual adaptation is built from per-task deltas (task vectors).
- During/after task training, KFAC statistics are used to approximate curvature terms for drift-aware regularization.
- At inference, merged vectors are applied over the visual backbone under the selected merging strategy.

## Limitations

- Artifact compatibility depends on matching dataset split/order and preprocessing assumptions.
- Fisher files are backend- and run-dependent; mixing incompatible runs can degrade results.
- This repository may contain research artifacts, not production-hardened models.

## Citation

```bibtex
@inproceedings{porrello2026dataless,
  title={Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature},
  author={Porrello, Angelo and Buzzega, Pietro and Dangel, Felix and Sommariva, Thomas and Salami, Riccardo and Bonicelli, Lorenzo and Calderara, Simone},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Resources

- Mammoth framework: https://github.com/aimagelab/mammoth
- TAK v2 implementation: `models/tak.py`
