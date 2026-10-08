from argparse import Namespace
import torch
from torch import nn
from models.dualprompt_utils.vision_transformer import vit_base_patch16_224_dualprompt
from models.l2p_utils.pretrained import checkpoint_spec, verify_pretrained_branches


class Model(nn.Module):
    def __init__(self, args: Namespace, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        size = getattr(args, 'dualprompt_vit_size', 'base')
        if size not in ('base', 'small', 'tiny'):
            raise ValueError('DualPrompt ViT size must be base, small or tiny')
        if size != 'base' and not getattr(args, 'pretrained_checkpoint', None):
            raise ValueError('Small/Tiny DualPrompt requires a local pretrained_checkpoint')
        self.vit_size = size
        checkpoint = getattr(args, 'pretrained_checkpoint', None)
        if checkpoint:
            spec = checkpoint_spec(size, checkpoint)
            checkpoint = spec['file']
        self.pretrained_checkpoint = checkpoint

        self.original_model = vit_base_patch16_224_dualprompt(
            vit_size=size,
            pretrained=args.pretrained,
            pretrained_checkpoint=checkpoint,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
            args=args
        )
        self.original_model.eval()

        self.model = vit_base_patch16_224_dualprompt(
            vit_size=size,
            pretrained=args.pretrained,
            pretrained_checkpoint=checkpoint,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
            prompt_length=args.length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=True,
            prompt_key=True,
            pool_size=args.size,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=True,
            use_g_prompt=True,
            g_prompt_length=args.g_prompt_length,
            g_prompt_layer_idx=args.g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
            use_e_prompt=True,
            e_prompt_layer_idx=args.e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
            same_key_value=args.same_key_value,
            args=args
        )

        if checkpoint:
            self.initial_canonical_pretrained_branches = verify_pretrained_branches(
                self.original_model, self.model, spec)
            self.initial_canonical_pretrained_sha256 = spec['canonical_sha256']
            self.pretrained_sha256 = spec['sha256']

        if args.freeze:
            for p in self.original_model.parameters():
                p.requires_grad = False

            for n, p in self.model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

    def forward(self, x, task_id, train=False, return_outputs=False):

        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=task_id, cls_features=cls_features, train=train)

        if return_outputs:
            return outputs
        else:
            return outputs['logits']
