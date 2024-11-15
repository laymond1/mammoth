"""
This is the base class for all models. It provides some useful methods and defines the interface of the models.

The `begin_task` and `end_task` methods are called before and after each task, respectively.

The `get_parser` method returns the parser of the model. Additional model-specific hyper-parameters can be added by overriding this method.

The `get_debug_iters` method returns the number of iterations to be used for debugging. Default: 3.

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
import logging
import os
import sys
import copy
import time
import datetime
from contextlib import suppress
from typing import Iterator, List, Tuple
import inspect
import numpy as np

import torch
import torch_optimizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn import Module
from torch.optim import lr_scheduler

from models.utils.continual_model import ContinualModel

from utils.magic import persistent_locals
from utils.metrics import calculate_online_forgetting

with suppress(ImportError):
    import wandb


class OnlineContinualModel(ContinualModel):
    """
        Online Continual learning model.
    """
    def __init__(self, backbone, loss, args, transform, dataset=None, **kwargs):
        super().__init__(backbone, loss, args, transform, dataset=dataset, **kwargs)        
        
        # self.ngpus_per_nodes = torch.cuda.device_count()
        self.ngpus_per_nodes = 1
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size = self.world_size * self.ngpus_per_nodes
        self.distributed = self.world_size > 1

        if self.distributed:
            self.args.batch_size = self.args.batch_size // self.world_size
        self.args.temp_batch_size = self.args.batch_size // 2       
        
        self.start_time = time.time()
        self.exposed_classes = []
        self.mask = torch.zeros(self.num_classes, device=self.device) - torch.inf
        self.seen = 0
        self.reset_opt()
        # attributes for evaluate metrics
        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []
        self.knowledge_loss_rate = []
        self.knowledge_gain_rate = []
        self.forgetting_time = []
        self.f_next_time = 0
        
    def add_new_class(self, class_name):
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.args.lr_scheduler: # Not used
            self.update_schedule(reset=True)
            
    def get_future_classes(self, data_loader):
        future_classes = []

        # Collect future classes from the data loader
        for _, labels, _, _ in data_loader:
            for label in labels:
                label_item = label.item()
                # If the class is not exposed, add it to the future_classes
                if label_item not in future_classes:
                    future_classes.append(label_item)

        return list(future_classes)

    def future_evaluate(self, test_loader, future_classes):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.num_classes)
        num_data_l = torch.zeros(self.num_classes)
        label = []
        
        # Create a mapping from label to index for future_classes
        class_to_idx = {label: idx for idx, label in enumerate(future_classes)}
        
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0], data[1]
                # Update y using the mapping
                for j in range(len(y)):
                    y[j] = class_to_idx[y[j].item()]

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.net(x, return_outputs=True)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = F.cross_entropy(logits, y)
                pred = torch.argmax(logits, dim=-1)
                _, _preds = logits.topk(1, 1, True, True) # self.topk: 1
                total_correct += torch.sum(_preds == y.unsqueeze(1)).item()
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

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.num_classes)
        num_data_l = torch.zeros(self.num_classes)
        label = []
        
        # Create a mapping from label to index for exposed_classes
        class_to_idx = {label: idx for idx, label in enumerate(self.exposed_classes)}
        
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0], data[1]
                # Update y using the mapping
                for j in range(len(y)):
                    y[j] = class_to_idx[y[j].item()]

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.net(x, return_outputs=True)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                logits = logits + self.mask
                loss = F.cross_entropy(logits, y)
                pred = torch.argmax(logits, dim=-1)
                _, _preds = logits.topk(1, 1, True, True) # self.topk: 1
                total_correct += torch.sum(_preds == y.unsqueeze(1)).item()
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
    
    def online_forgetting_evaluate(self, test_loader, future_classes, samples_cnt):
        preds = []
        gts = []
        
        # Create a mapping from label to index for future_classes
        class_to_idx = {label: idx for idx, label in enumerate(future_classes)}

        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0], data[1]
                # Update y using the mapping
                for j in range(len(y)):
                    y[j] = class_to_idx[y[j].item()]

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.net(x, return_outputs=True)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                logits = logits + self.mask
                pred = torch.argmax(logits, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
            
            preds = np.concatenate(preds)
            if self.gt_label is None:
                self.gt_label = np.concatenate(gts)

        # Combine get_forgetting logic here
        self.test_records.append(preds)
        self.n_model_cls.append(copy.deepcopy(len(self.exposed_classes)))
        
        # Initialize klr and kgr with default values
        klr, kgr = 0.0, 0.0
        
        if len(self.test_records) > 1:
            klr, kgr = calculate_online_forgetting(
                len(future_classes),
                self.gt_label, 
                self.test_records[-2], 
                self.test_records[-1], 
                self.n_model_cls[-2], 
                self.n_model_cls[-1]
            )
            self.knowledge_loss_rate.append(klr)
            self.knowledge_gain_rate.append(kgr)
            self.forgetting_time.append(samples_cnt)
            np.save(f"{self.args.log_path}/logs/{self.args.online_scenario}/{self.args.dataset}/{self.args.model}/KLR_seed_{self.args.seed}.npy", self.knowledge_loss_rate)
            np.save(f"{self.args.log_path}/logs/{self.args.online_scenario}/{self.args.dataset}/{self.args.model}/KGR_seed_{self.args.seed}.npy", self.knowledge_gain_rate)
            np.save(f"{self.args.log_path}/logs/{self.args.online_scenario}/{self.args.dataset}/{self.args.model}/forgetting_time_seed_{self.args.seed}.npy", self.forgetting_time)
        
        fgt_eval_dict = {"klr": klr, "kgr": kgr}
        
        return fgt_eval_dict

    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_train(self, data):
        raise NotImplementedError()

    def model_forward(self, x, y):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_before_train(self):
        raise NotImplementedError()
    
    def online_after_task(self, task_id):
        raise NotImplementedError()

    def online_after_train(self):
        raise NotImplementedError()
    
    # def online_evaluate(self, test_loader, samples_cnt):
    #     raise NotImplementedError()
            
    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0
    
    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def report_training(self, total_samples, sample_num, train_loss_dict, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss_dict['total_loss']:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            # Add counts of each prompt if available
            + (f"Prompt Counts {self.net.train_count.to(torch.int64).tolist()} | " if hasattr(self.net, 'train_count') else "") +
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (total_samples - sample_num) / sample_num))}"
        )

        if 'wandb' in sys.modules and not self.args.nowand:
            self.online_train_autolog_wandb(sample_num, train_loss_dict, train_acc)

    def report_test(self, sample_num, eval_dict, other_metric=None, task_id=None):
        avg_loss = eval_dict.get("avg_loss", 0.0)
        avg_acc = eval_dict.get("avg_acc", 0.0)
        klr = eval_dict.get("klr", None)
        kgr = eval_dict.get("kgr", None)
    
        # Construct the base print message
        message = f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        
        # Add other metrics if they exist
        if other_metric is not None:
            message += f"instant_fgt {other_metric.get('instant_fgt', 0.0):.4f} | last_fgt {other_metric.get('last_fgt', 0.0):.4f} | "
        
        # Add klr and kgr to the message if they exist
        if klr is not None:
            message += f"klr {klr:.4f} | "
        if kgr is not None:
            message += f"kgr {kgr:.4f} | "

        # Print the message
        if task_id is None:
            print(message)
        else:
            print(f"Task {task_id}: {message}")
            
        # Log to wandb if applicable
        if 'wandb' in sys.modules and not self.args.nowand and task_id is None:
            self.online_test_autolog_wandb(sample_num, eval_dict, extra=other_metric)
                
    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.num_classes)
        ret_corrects = torch.zeros(self.num_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.args.optimizer, self.args.lr, self.get_parameters())
        self.scheduler = select_scheduler(self.args.lr_scheduler, self.optimizer)
    
    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.args.lr_scheduler, self.optimizer, self.args.sched_multistep_lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.args.lr
        else:
            self.scheduler.step()
            
    def online_train_autolog_wandb(self, sample_num, train_loss_dict, train_acc, prefix='Train', extra=None):
        """
        All variables starting with "_wandb_" or "loss_dict" or "acc" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            tmp = {'# of samples': sample_num,
                   f'{prefix}_acc': train_acc,
                   'num_classes': len(self.exposed_classes)}
            # add counts of each prompt
            if hasattr(self, 'table'):
                self.table.add_data(sample_num, *self.net.train_count.to(torch.int64).tolist())
            tmp.update({f'{prefix}_{k}': v for k, v in train_loss_dict.items()})
            tmp.update(extra or {})
            if hasattr(self, 'opt'):
                tmp['lr'] = self.opt.param_groups[0]['lr']
            wandb.log(tmp, step=sample_num)
    
    def online_test_autolog_wandb(self, sample_num, eval_dict, prefix='Test', extra=None):
        """
        All variables starting with "_wandb_" or "loss" or "acc" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            tmp = {'# of samples': sample_num,
                f'{prefix}_acc': eval_dict.get("avg_acc", 0.0),
                f'{prefix}_loss': eval_dict.get("avg_loss", 0.0),
                'num_classes': len(self.exposed_classes)}
            # Add klr and kgr if they exist
            if eval_dict.get("klr") is not None:
                tmp['klr'] = eval_dict["klr"]
            if eval_dict.get("kgr") is not None:
                tmp['kgr'] = eval_dict["kgr"]
            # Add any other metrics from the extra parameter
            tmp.update(extra or {})
            wandb.log(tmp, step=sample_num)
            
        
def select_optimizer(opt_name: str, lr: float, params: Iterator[torch.Tensor]) -> optim.Optimizer:
    if opt_name == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt


def select_scheduler(lr_scheduler: str, opt: optim.Optimizer, hparam=None) -> lr_scheduler._LRScheduler:
    if "exp" in lr_scheduler:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif lr_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif lr_scheduler == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif lr_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif lr_scheduler == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1) # default
    return scheduler