"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.online_continual_model import OnlineContinualModel
from models.prompt_utils.model import PromptModel
from utils.buffer import Buffer

import wandb


class OnlineL2P(OnlineContinualModel):
    """Learning to Prompt (L2P)."""
    NAME = 'online-l2p'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian'] # sdp, stream

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Replay parameters
        add_rehearsal_args(parser)
        # Trick
        parser.add_argument('--train_mask', type=int, default=1, choices=[0, 1], help='if using the class mask at training')

        # Prompt parameters
        parser.add_argument('--e_prompt_pool_size', type=int, default=30, help='number of prompts (M in paper)')
        parser.add_argument('--e_prompt_length', type=int, default=20, help='length of prompt (L_p in paper)')
        parser.add_argument('--top_k', type=int, default=5, help='top k prompts to use (N in paper)')
        parser.add_argument('--pull_constraint_coeff', type=float, default=0.5, help='Coefficient for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')
        parser.add_argument('--same_key_value', type=bool, default=False, help='the same key-value across all layers of the E-Prompt')

        # Prompt location
        parser.add_argument('--shallow', type=bool, default=True, help='Shallow ViT vs. Deep ViT')
        
        # ETC
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        L2P re-defines the backbone model to include the prompt parameters. This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        del backbone
        print("-" * 20)
        print(f"WARNING: L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        if not args.shallow:
            args.e_prompt_length = 5

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='l2p',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # buffer 
        self.buffer = Buffer(self.args.buffer_size)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        # init task per class
        self.task_per_cls = [0]
    
    def online_before_task(self, task_id):
        if task_id > 0:
            self.net.prompt.process_task_count()
        self.subset_start = self.task_per_cls[task_id]
        pass

    def online_before_train(self):
        pass
    
    def online_step(self, inputs, labels, not_aug_inputs, idx):
        self.add_new_class(labels)
        
        _loss_dict = dict()
        _acc, _iter = 0.0, 0
        real_batch_size = inputs.shape[0]

        # sample data from the buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        for _ in range(int(self.args.online_iter)):
            loss_dict, acc = self.online_train([inputs.clone(), labels.clone()])
            _loss_dict = {k: v + _loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            _acc += acc
            _iter += 1
        if self.args.buffer_size > 0:
            # add new data to the buffer
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels[:real_batch_size])

        del(inputs, labels)
        gc.collect()
        
        _loss_dict = {k: v / _iter for k, v in _loss_dict.items()}
        return _loss_dict, _acc / _iter
    
    def online_train(self, data):
        self.net.train()
        total_loss_dict = dict()
        total_correct, total_num_data = 0.0, 0.0
        class_to_idx = {label: idx for idx, label in enumerate(self.exposed_classes)}

        x, y = data
        
        for j in range(len(y)):
            y[j] = class_to_idx[y[j].item()]

        x = x.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        logits, loss_dict = self.model_forward(x, y) 
        loss = loss_dict['total_loss']
        _, preds = logits.topk(1, 1, True, True) # self.topk: 1
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss_dict = {k: v + total_loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss_dict, total_correct/total_num_data
    
    def model_forward(self, x, y):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            logits, prompt_loss = self.net(x, train=True)
            loss_dict = dict()
            # here is the trick to mask out classes of non-current classes
            non_cur_classes_mask = torch.zeros(self.num_classes, device=self.device) - torch.inf
            non_cur_classes_mask[y.unique()] = 0
            # mask out unseen classes and non-current classes
            if self.args.train_mask:
                logits = logits + non_cur_classes_mask
            else:
                logits = logits + self.mask
            
            ce_loss = self.loss(logits, y)
            if self.args.pull_constraint_coeff > 0.0:
                cos_loss = self.args.pull_constraint_coeff * prompt_loss
                total_loss = ce_loss + cos_loss
                loss_dict.update({'ce_loss': ce_loss})
                loss_dict.update({'cos_loss': cos_loss})
                loss_dict.update({'total_loss': total_loss})
            else:
                loss_dict.update({'total_loss': ce_loss})
                
        return logits, loss_dict

    def online_after_task(self, task_id):
        self.task_per_cls.append(len(self.exposed_classes))
        pass

    def online_after_train(self):
        pass

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]