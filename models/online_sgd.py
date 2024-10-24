"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import torch
import torch.nn.functional as F

from datasets import get_dataset
from models.vit_utils.vit_model import ViTModel
from models.utils.online_continual_model import OnlineContinualModel
from utils.args import ArgumentParser


class OnlineSgd(OnlineContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'online-sgd'
    COMPATIBILITY = ['online-il']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        # Trick parameters
        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        logging.info(f"SGD USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)
        
        backbone = ViTModel(args)
        
        super(OnlineSgd, self).__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)

    def online_before_task(self, task_id):
        pass

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def online_step(self, inputs, labels, idx):
        self.add_new_class(labels)
        _loss_dict = dict()
        _acc, _iter = 0.0, 0

        for _ in range(int(self.args.online_iter)):
            loss_dict, acc = self.online_train([inputs.clone(), labels.clone()])
            _loss_dict = {k: v + _loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            _acc += acc
            _iter += 1
        del(inputs, labels)
        gc.collect()
        _loss_dict = {k: v / _iter for k, v in _loss_dict.items()}
        return _loss_dict, _acc / _iter

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
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.update_schedule()

        total_loss_dict = {k: v + total_loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss_dict, total_correct/total_num_data
    
    def model_forward(self, x, y):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            outputs = self.net(x, return_outputs=True)
            logits = outputs['logits']
            loss_dict = dict()
            # mask out unseen classes
            logits = logits + self.mask
            
            ce_loss = self.loss(logits, y)
            loss_dict.update({'total_loss': ce_loss})
                
        return logits, loss_dict

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

                outputs = self.net(x, return_outputs=True)
                logits = outputs['logits']
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