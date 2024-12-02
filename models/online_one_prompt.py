"""
OVOR: OnePrompt

Note:
    OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import time
import datetime
import logging
import torch
import torch.nn.functional as F

from models.oneprompt_utils.ood import NPOS
from models.oneprompt_utils.oneprompt_model import OnePromptModel
from models.utils.online_continual_model import OnlineContinualModel
from utils.args import *

from datasets import get_dataset

import wandb


class OnlineOnePrompt(OnlineContinualModel):
    """Continual Learning via OnePrompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'online-one-prompt'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--e_prompt_pool_size', type=int, default=10, help='size of E-Prompt pool')
        parser.add_argument('--e_prompt_length', type=int, default=40, help='length of E-Prompt')
        parser.add_argument('--g_prompt_length', type=int, default=10, help='length of G-Prompt')
        # parser.add_argument('--prompt_param', nargs='+', type=float, default=[10, 40, 10],
        #                 help='e prompt pool size, e prompt length, g prompt length')
        # Optimizer parameters
        parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
        # Trick parameters
        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        logging.info(f"OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        if args.lr_scheduler is not None:
            logging.info("OnePrompt uses a custom scheduler: cosine. Ignoring --lr_scheduler.")

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        # n_tasks = args.n_tasks # no task boundary
        num_classes =tmp_dataset.N_CLASSES
        
        backbone = OnePromptModel(num_classes=num_classes,
                                  pretrained=True,
                                  prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.g_prompt_length])
        
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)

        self.ood = NPOS(args)

    def get_optimizer(self):
        params_to_opt = list(self.net.prompt.parameters()) + list(self.net.last.parameters())
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.args.lr,
                         'weight_decay': self.args.optim_wd}
        if self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(**optimizer_arg)
        elif self.args.optimizer == 'adam':
            opt = torch.optim.Adam(**optimizer_arg)
        else:
            raise ValueError('Optimizer not supported for this method')
        return opt
            
    def online_before_train(self):
        pass
        
    def online_step(self, inputs, labels, idx):
        self.add_new_class(labels)
        _loss_dict = dict()
        _ood_loss_dict = dict()
        _acc, _ood_acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.args.online_iter)):
            loss_dict, acc = self.online_train([inputs.clone(), labels.clone()])
            ood_loss_dict, ood_acc = self.online_train_ood([inputs.clone(), labels.clone()])
            _loss_dict = {k: v + _loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            _ood_loss_dict = {k: v + _ood_loss_dict.get(k, 0.0) for k, v in ood_loss_dict.items()}
            _acc += acc
            _ood_acc += ood_acc
            _iter += 1
        del(inputs, labels)
        gc.collect()
        
        _loss_dict = {k: v / _iter for k, v in _loss_dict.items()}
        _ood_loss_dict = {k: v / _iter for k, v in _ood_loss_dict.items()}
        return _loss_dict, _ood_loss_dict, _acc / _iter, _ood_acc / _iter
    
    def online_train(self, data):
        self.net.train()
        total_loss_dict = dict()
        total_correct, total_num_data = 0.0, 0.0

        x, y = data
        self.labels = torch.cat((self.labels, y), 0)
        
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)
        
        self.opt.zero_grad()
        logits, loss_dict = self.model_forward(x, y) 
        loss = loss_dict['total_loss']
        _, preds = logits.topk(1, 1, True, True) # self.topk: 1
               
        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.scaler.step(self.opt)
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
        total_correct, total_num_data = 0.0, 0.0

        x, y = data
        self.labels = torch.cat((self.labels, y), 0)
        
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

        self.opt.zero_grad()
        logits, loss_dict = self.model_ood_forward(ids_x, oods_x, targets)
        loss = loss_dict['total_loss']
        _, preds = logits.topk(1, 1, True, True) # self.topk: 1

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.scaler.step(self.opt)
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
            prompt_loss = prompt_loss.sum()
            total_loss = ce_loss + prompt_loss
            
            loss_dict.update({'ce_loss': ce_loss})
            loss_dict.update({'prompt_loss': prompt_loss})
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

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.num_classes)
        num_data_l = torch.zeros(self.num_classes)
        label = []
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0], data[1]
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x, train=False)
                logits = logits + self.mask
                loss = F.cross_entropy(logits, y)
                pred = torch.argmax(logits, dim=-1)
                _, preds = logits.topk(1, 1, True, True) # self.topk: 1
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict
    
    def online_after_task(self, task_id):
        pass
    
    def online_after_train(self):
        pass
    
    def get_parameters(self):
        return list(self.net.prompt.parameters()) + list(self.net.last.parameters())
    
    def report_training(self, total_samples, sample_num, train_loss_dict, train_acc, train_ood_loss_dict, train_ood_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss_dict['total_loss']:.4f} | train_acc {train_acc:.4f} | "
            f"train_ood_loss {train_ood_loss_dict['total_loss']:.4f} | train_ood_acc {train_ood_acc:.4f} |  train_id_score {train_ood_loss_dict['id_score']:.4f} |  train_ood_score {train_ood_loss_dict['ood_score']:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            # Add counts of each prompt if available
            + (f"Prompt Counts {self.net.train_count.to(torch.int64).tolist()} | " if hasattr(self.net, 'train_count') else "") +
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
            # add counts of each prompt
            if hasattr(self, 'table'):
                self.table.add_data(sample_num, *self.net.train_count.to(torch.int64).tolist())
            tmp.update({f'{prefix}_{k}': v for k, v in train_loss_dict.items()})
            tmp.update({f'{prefix}_ood_{k}': v for k, v in train_ood_loss_dict.items()})
            tmp.update(extra or {})
            if hasattr(self, 'opt'):
                tmp['lr'] = self.opt.param_groups[0]['lr']
            wandb.log(tmp, step=sample_num)