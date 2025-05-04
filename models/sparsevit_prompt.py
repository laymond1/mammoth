"""
Sparse Prompt Learning for On-Device Continual Learning

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.continual_model import ContinualModel
from models.sparsevit_prompt_utils.model import PromptModel
# from models.prompt_utils.model_quantized import PromptModel
from utils.schedulers import CosineSchedule
from utils.buffer import Buffer
from utils import parse_str_to_int, binary_to_boolean_type

import wandb


class SparseViTPrompt(ContinualModel):
    """Continual Learning via CODA-Prompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'sparsevit-prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--sparse_type', type=str, default='random', choices=['random', 'l2', 'attn_map'], help='sparse update type')
        parser.add_argument('--drop_rate', type=float, default=0.3, help='Token purning ratio')
        # parser.add_argument('--query', type=str, default='poolformer', choices=['vit', 'poolformer'], help="choose one of [poolformer]")
        # parser.add_argument('--e_prompt_layer_idx', type=int, default=[-5, -4, -3, -2, -1], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[0, 1, 2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--e_prompt_length', type=int, default=8, help='prompt length')
        parser.add_argument('--ortho_mu', type=float, default=0.0, help='orthogonal penalty weight') # but it's set to 0.0 becuase of (#issue12)[https://github.com/GT-RIPL/CODA-Prompt/issues/12]
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient(mu) for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')
        parser.add_argument('--same_key_value', type=bool, default=False, help='the same key-value across all layers of the E-Prompt')
        parser.add_argument('--n_splits', type=int, default=None, help='Number of splits for the prompt pool (default: 1).')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        args.n_tasks = tmp_dataset.N_TASKS
        if args.n_splits is not None:
            assert num_classes % args.n_splits == 0
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='sparse',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.ortho_mu])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
    
    def begin_task(self, dataset):
        if self.current_task > 0:
            self.net.prompt.process_task_count()
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        if self.args.use_scheduler:
            self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_epoch(self, epoch, dataset):
        self.count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
        logits, loss_prompt = self.net(inputs, train=True)
        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint_coeff > 0.0 and loss_prompt is not None:
            loss = loss + self.args.pull_constraint_coeff * loss_prompt.mean()  # the mean is needed for data-parallel (concatenates instead of averaging)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()
        # self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        # self.scaler.step(self.opt)
        # self.scaler.update()

        # Calculate accuracy
        preds = torch.argmax(logits[:, :self.n_seen_classes], dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        # Update running loss와 accuracy
        self.count += 1
        self.running_loss += loss.item()
        self.running_accuracy += accuracy

        return loss.item()
    
    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]