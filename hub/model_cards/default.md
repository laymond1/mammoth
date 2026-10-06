---
language: en
license: {{LICENSE}}
library_name: pytorch
tags:
- continual-learning
- mammoth
pipeline_tag: image-classification
---

# {{MODEL_NAME_DISPLAY}}

This repository hosts artifacts for the **{{MODEL_NAME_DISPLAY}}** model used with the Mammoth continual-learning framework.

## Model description

{{MODEL_NAME_DISPLAY}} is a continual-learning method integrated in Mammoth as `--model {{MODEL_ID}}`.

This Hub repository can contain:

- model checkpoints,
- auxiliary tensors (for example, Fisher/KFAC statistics),
- reproducibility metadata and run commands.

## Intended use

- Research and reproducibility in continual learning.
- Evaluation with Mammoth datasets and settings.

## How to use

Install Mammoth dependencies and run:

```bash
uv run python main.py --model {{MODEL_ID}} --dataset <dataset-name> --model_config best
```

If this repository includes external artifacts, download/load them according to the model documentation.

## Repository layout

Common layout used in this repo:

- `README.md`: model card
- `artifacts/` or task-specific folders: uploaded tensors/checkpoints

## Citation

If you use this model repository, please cite:

- Mammoth: https://github.com/aimagelab/mammoth
- The original paper for {{MODEL_NAME_DISPLAY}} (when available)

## Contact

- Hub repo: https://huggingface.co/{{REPO_ID}}
- Framework: https://github.com/aimagelab/mammoth
