"""
OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning

Note:
    OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import sys
import gc
import time
import datetime
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.online_continual_model import OnlineContinualModel
from models.prompt_utils.model import PromptModel
from models.oneprompt_utils.ood import NPOS
from utils.buffer import Buffer

import wandb


class OnlineOVOR(OnlineContinualModel):
    """OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning."""
    NAME = 'online-ovor'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Replay parameters
        add_rehearsal_args(parser)
        # Trick
        parser.add_argument('--train_mask', type=int, default=1, choices=[0, 1], help='if using the class mask at training')

        # G-Prompt parameters
        parser.add_argument('--g_prompt_layer_idx', type=int, default=[0, 1], nargs="+", help='the layer index of the G-Prompt')
        parser.add_argument('--g_prompt_length', type=int, default=10, help='length of G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=1, help='number of prompts (fixed: 1)')
        parser.add_argument('--e_prompt_length', type=int, default=40, help='length of E-Prompt')
        
        # OOD parameters
        parser.add_argument('--cov', type=float, default=1.0) # 0.1 for CUB200
        parser.add_argument('--thres_id', type=float, default=-24.0) # -15.0 for ImageNet-A
        parser.add_argument('--thres_ood', type=float, default=-3.0)
        parser.add_argument('--num_per_class', type=int, default=40)
        parser.add_argument('--sample_from', type=int, default=600)
        parser.add_argument('--select', type=int, default=50)
        parser.add_argument('--pick_nums', type=int, default=30)
        parser.add_argument('--K', type=int, default=50)
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--huber', action='store_false')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='oneprompt',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.g_prompt_length])
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # buffer 
        self.buffer = Buffer(self.args.buffer_size)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.ood = NPOS(args)
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
        _ood_loss_dict = dict()
        _acc, _ood_acc, _iter = 0.0, 0.0, 0

        real_batch_size = inputs.shape[0]

        # sample data from the buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        for i in range(int(self.args.online_iter)):
            loss_dict, acc = self.online_train([inputs.clone(), labels.clone()])
            _loss_dict = {k: v + _loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            _acc += acc
            _iter += 1

            # the last iter
            if i == int(self.args.online_iter) - 1:
                ood_loss_dict, ood_acc = self.online_train_ood([inputs.clone(), labels.clone()])
                _ood_loss_dict = {k: v + _ood_loss_dict.get(k, 0.0) for k, v in ood_loss_dict.items()}
                _ood_acc += ood_acc
            
        if self.args.buffer_size > 0:
            # add new data to the buffer
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels[:real_batch_size])

        del(inputs, labels)
        gc.collect()
        
        _loss_dict = {k: v / _iter for k, v in _loss_dict.items()}
        _ood_loss_dict = {k: v / 1 if _ood_loss_dict else 0.0 for k, v in _ood_loss_dict.items()}  # ood 값은 마지막 iter만 사용
        return _loss_dict, _ood_loss_dict, _acc / _iter, _ood_acc  # ood acc는 단일 값
    
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
    
    def _get_ood_samples(self, x, y):
        sorted_y, sorted_idx = torch.sort(y.detach().cpu())
        with torch.no_grad():
            feats = self.net(x, train=False, feat=True).detach().cpu()
            id_feats = feats[sorted_idx]

        return self.ood.generate(id_feats, sorted_y)

    def online_train_ood(self, data):
        self.net.train()
        total_loss_dict = dict()
        total_correct, total_num_data, _iter = 0.0, 0.0, 0

        x, y = data
        
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)
        # get ood samples (features)
        id_loader, ood_loader = self._get_ood_samples(x, y)
        # print("length of id_loader: ", len(id_loader))
        # print("length of ood_loader: ", len(ood_loader))
        for ((ids_x, targets), oods) in zip(id_loader, ood_loader):
            oods_x = oods[0]

            ids_x = ids_x.to(self.device)
            oods_x = oods_x.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits, loss_dict = self.model_ood_forward(ids_x, oods_x, targets)
            loss = loss_dict['total_loss']
            _, preds = logits.topk(1, 1, True, True) # self.topk: 1

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_schedule()

            total_loss_dict = {k: v + total_loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            total_correct += torch.sum(preds == targets.unsqueeze(1)).item()
            total_num_data += targets.size(0)
            _iter += 1

        total_loss_dict = {k: v / _iter for k, v in total_loss_dict.items()}

        return total_loss_dict, total_correct/total_num_data
    
    def model_forward(self, x, y):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            logits, _ = self.net(x, train=True)
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
            total_loss = ce_loss
            loss_dict.update({'ce_loss': ce_loss})
            loss_dict.update({'total_loss': total_loss})

        return logits, loss_dict
        
    def model_ood_forward(self, id_x, ood_x, y):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            id_logits = self.net(id_x, last=True)
            ood_logits = self.net(ood_x, last=True)
            loss_dict = dict()
            # here is the trick to mask out classes of non-current classes
            non_cur_classes_mask = torch.zeros(self.num_classes, device=self.device) - torch.inf
            non_cur_classes_mask[y.unique()] = 0 
            # mask out unseen classes and non-current classes
            if self.args.train_mask:
                id_logits = id_logits + non_cur_classes_mask
                ood_logits = ood_logits + non_cur_classes_mask
            else:
                id_logits = id_logits + self.mask
                ood_logits = ood_logits + self.mask
            
            ce_loss = self.loss(id_logits, y)
            ood_loss, id_score, ood_score = self.ood.loss(id_logits, ood_logits)
            total_loss = ce_loss + ood_loss
            
            loss_dict.update({'id_ce_loss': ce_loss})
            loss_dict.update({'ood_loss': ood_loss})
            loss_dict.update({'id_score': id_score})
            loss_dict.update({'ood_score': ood_score})
            loss_dict.update({'total_loss': total_loss})
            
        return id_logits, loss_dict
    
    def online_after_task(self, task_id):
        self.task_per_cls.append(len(self.exposed_classes))
        pass

    def online_after_train(self):
        pass

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def report_training(self, total_samples, sample_num, train_loss_dict, train_acc, train_ood_loss_dict, train_ood_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss_dict['total_loss']:.4f} | train_acc {train_acc:.4f} | "
            f"train_ood_loss {train_ood_loss_dict['total_loss']:.4f} | train_ood_acc {train_ood_acc:.4f} |  train_id_score {train_ood_loss_dict['id_score']:.4f} |  train_ood_score {train_ood_loss_dict['ood_score']:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            # Add counts of each prompt if available
            + (f"Prompt Counts {self.net.prompt.train_count.to(torch.int64).tolist()} | " if hasattr(self.net, 'train_count') else "") +
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (total_samples - sample_num) / sample_num))}"
        )

        if 'wandb' in sys.modules and not self.args.nowand:
            self.online_train_autolog_wandb(sample_num, train_loss_dict, train_ood_loss_dict, train_acc, train_ood_acc)

    def online_train_autolog_wandb(self, sample_num, train_loss_dict, train_ood_loss_dict, train_acc, train_ood_acc, prefix='Train', extra=None):
        """
        All variables starting with "_wandb_" or "loss_dict" or "acc" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            tmp = {'# of samples': sample_num,
                   f'{prefix}_acc': train_acc,
                   f'{prefix}_ood_acc': train_ood_acc,
                   'num_classes': len(self.exposed_classes)}
            tmp.update({f'{prefix}_{k}': v for k, v in train_loss_dict.items()})
            tmp.update({f'{prefix}_ood_{k}': v for k, v in train_ood_loss_dict.items()})
            tmp.update(extra or {})
            if hasattr(self, 'opt'):
                tmp['lr'] = self.optimizer.param_groups[0]['lr']
            wandb.log(tmp, step=sample_num)