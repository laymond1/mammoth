"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

from collections import defaultdict
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.continual_model import ContinualModel
from models.prompt_utils.model import PromptModel
# from models.prompt_utils.model_quantized import PromptModel
from utils.buffer import Buffer
from models.prompt_utils.protos import Prototypes

import wandb


class ProtoCodaPrompt(ContinualModel):
    """Continual Learning via CODA-Prompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'proto-coda-prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--e_prompt_pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--e_prompt_length', type=int, default=8, help='prompt length')
        parser.add_argument('--ortho_mu', type=float, default=0.0, help='orthogonal penalty weight') # but it's set to 0.0 becuase of (#issue12)[https://github.com/GT-RIPL/CODA-Prompt/issues/12]
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient(mu) for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')
        parser.add_argument('--same_key_value', type=bool, default=False, help='the same key-value across all layers of the E-Prompt')
        parser.add_argument('--n_splits', type=int, default=None, help='Number of splits for the prompt pool (default: 1).')

        # Prototype
        parser.add_argument('--proto_trans', type=bool, default=True, help='Use prototype transformation')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')

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
                               pretrained=True, prompt_flag='coda',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.ortho_mu])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.prototypes = Prototypes(args, self.net.feat.embed_dim, self.device)
        
    
    def begin_task(self, dataset):
        if self.current_task > 0:
            self.net.prompt.process_task_count()
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        self.query_buffer = torch.zeros((len(dataset.train_loader.dataset), self.net.feat.embed_dim)).to(self.device)
        self.indexes_memory = defaultdict(list)

    def begin_epoch(self, epoch, dataset):
        self.count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.indexes = []

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, indexes=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device
        
        if epoch == 0:
            logits, loss_prompt = self.train_epoch_1(inputs, labels, not_aug_inputs, epoch, indexes)
        else:
            logits, loss_prompt = self.train_epoch_rest(inputs, labels, not_aug_inputs, epoch, indexes)

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

        # 
        self.indexes.extend(indexes.cpu().numpy().tolist())

        return loss.item()
    
    def end_epoch(self, epoch, dataset):
        self.indexes_memory[epoch] = self.indexes
    
    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
    
    def train_epoch_1(self, inputs, labels, not_aug_inputs, epoch=None, indexes=None):
        # query를 buffer에 저장 
        with torch.no_grad():
            q, _ = self.net.feat(inputs)
            q = q[:, 0, :]
        # self.prototypes.update(q, labels)

        # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
        logits, loss_prompt = self.net(inputs, q=q, train=True)
        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        # to avoid device inconsistency
        for i in range(len(indexes)):
            idx = indexes[i].item()
            self.query_buffer[idx] = q[i].clone()
        return logits, loss_prompt

    def train_epoch_rest(self, inputs, labels, not_aug_inputs, epoch=None, indexes=None):
        # query buffer에서 query를 가져옴 
        # aug_protos, _ = self.prototypes.generate_proto_data(labels, self.n_seen_classes)
        q = self.query_buffer[indexes].to(self.device)

        # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
        logits, loss_prompt = self.net(inputs, q=q, train=True)
        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        return logits, loss_prompt
    
    # def end_epoch(self, epoch, dataset=None):
        # proto_mean, proto_vars = self.prototypes.get_global_prototypes()
    

