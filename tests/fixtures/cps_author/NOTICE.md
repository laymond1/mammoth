# Third-Party Notices

This project (CPS-Prompt) is licensed under the MIT License (see `LICENSE`).
It incorporates and/or is derived from the following third-party works, each of
which retains its original license.

## Mammoth (MIT License)

This project is built on top of the Mammoth continual-learning framework.
Most files under `utils/`, `datasets/`, `backbone/`, `models/utils/`, and the
top-level `main.py` originate from or are adapted from Mammoth.

- Repository: <https://github.com/aimagelab/mammoth>
- License: MIT

## CODA-Prompt (MIT License)

The prompt-pool implementation and the BLIP-derived ViT used by `PromptModel`
are adapted from CODA-Prompt:

- `models/prompt_utils/prompt.py`
- `models/prompt_utils/vit.py`

Repository: <https://github.com/GT-RIPL/CODA-Prompt>
License: MIT

## timm Vision Transformer (Apache 2.0)

The following file is a trimmed/adapted version of the Vision Transformer
implementation from `timm` and is distributed under the Apache License 2.0:

- `backbone/vit.py`

Repository: <https://github.com/huggingface/pytorch-image-models>
License: Apache License, Version 2.0
