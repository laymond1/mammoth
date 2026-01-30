"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.continual_model import ContinualModel
from models.prompt_utils.model import PromptModel
from utils.schedulers import CosineSchedule
from utils.buffer import Buffer
from utils import parse_str_to_int, binary_to_boolean_type

import wandb


class CodaPrompt(ContinualModel):
    """Continual Learning via CODA-Prompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'coda-prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--query', type=str, default='vit', choices=['vit', 'poolformer'], help="choose one of [poolformer]")
        parser.add_argument('--e_prompt_layer_idx', type=parse_str_to_int, default=[0, 1, 2, 3, 4], help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--e_prompt_length', type=int, default=8, help='prompt length')
        parser.add_argument('--ortho_mu', type=float, default=0.0, help='orthogonal penalty weight') # but it's set to 0.0 becuase of (#issue12)[https://github.com/GT-RIPL/CODA-Prompt/issues/12]
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient(mu) for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')
        parser.add_argument('--same_key_value', type=bool, default=False, help='the same key-value across all layers of the E-Prompt')
        parser.add_argument('--head_epoch_start_ratio', type=float, default=1.0, help='the ratio of the epochs to start training the head')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_amp_opt', type=binary_to_boolean_type, default=False, help='Use automatic mixed precision')
        parser.add_argument('--use_grad_checkpoint', type=binary_to_boolean_type, default=False,
                            help='Use activation checkpointing per block in ViT')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')
        parser.add_argument('--peak_mem_log_ratio', type=float, default=0.0,
                            help='Ratio (0~1) of iterations per epoch to log peak memory right after forward')

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
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='coda',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.ortho_mu])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp_opt)
        if getattr(self.args, 'use_grad_checkpoint', False):
            self.net.feat.set_grad_checkpointing(True)
    
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
        # per-epoch window for peak memory logging
        if self.args.peak_mem_log_ratio > 0.0:
            if hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
                self._peak_mem_log_budget = max(1, int(self.args.peak_mem_log_ratio * len(dataset.train_loader)))
            else:
                self._peak_mem_log_budget = 1
        else:
            self._peak_mem_log_budget = 0
        self._peak_mem_logged = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        do_peak_log = (self._peak_mem_log_budget > 0
                       and self._peak_mem_logged < self._peak_mem_log_budget
                       and torch.cuda.is_available()
                       and device.type == 'cuda')
        if do_peak_log:
            torch.cuda.reset_peak_memory_stats(device)

        if self.args.use_amp_opt:
            with torch.amp.autocast(device_type=device.type, enabled=True):
                if epoch < int(self.args.n_epochs * self.args.head_epoch_start_ratio):
                    logits, loss_prompt = self.net(inputs, train=True)
                else:
                    with torch.no_grad():
                        feats = self.net(inputs, feat=True, train=False).detach()
                    logits = self.net(feats, last=True)
                    loss_prompt = None
        else:
            if epoch < int(self.args.n_epochs * self.args.head_epoch_start_ratio):
                logits, loss_prompt = self.net(inputs, train=True)
            else:
                with torch.no_grad():
                    feats = self.net(inputs, feat=True, train=False).detach()
                logits = self.net(feats, last=True)
                loss_prompt = None

        if do_peak_log:
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            cur_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            if not self.args.nowand and not self.args.debug_mode:
                wandb.log({'peak_mem_mb_forward': peak_mem_mb,
                           'cur_mem_mb_forward': cur_mem_mb})
            self._peak_mem_logged += 1
        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint_coeff > 0.0 and loss_prompt is not None:
            loss = loss + self.args.pull_constraint_coeff * loss_prompt.mean()  # the mean is needed for data-parallel (concatenates instead of averaging)

        self.opt.zero_grad(set_to_none=True)

        if self.args.use_amp_opt:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            grad_total_norm = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
            self.opt.step()

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
