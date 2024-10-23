import torch
import torch.nn as nn
from datasets import get_dataset
from models.l2p_utils.vit_prompt import vit_base_patch16_224_l2p


class ViTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        dataset = get_dataset(args)
        n_classes = dataset.N_CLASSES

        self.model = vit_base_patch16_224_l2p(
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )

    def forward(self, x, return_outputs=False):
        outputs = self.model(x)

        logits = outputs['logits']
        if return_outputs:
            return outputs
        else:
            return logits
