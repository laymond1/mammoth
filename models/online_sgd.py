"""
This module implements the simplest form of incremental training, i.e., finetuning.

Additionally online-sgd is modified for online continual learinng.

Note: 
    Online-SGD USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import torch

from datasets import get_dataset
from utils.args import ArgumentParser

from models.utils.online_continual_model import OnlineContinualModel
from models.prompt_utils.model import PromptModel


class OnlineSgd(OnlineContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'online-sgd'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Trick
        parser.add_argument('--train_mask', type=int, default=1, choices=[0, 1], help='if using the class mask at training')
        # Backbone
        parser.add_argument('--ft_backbone', type=bool, default=0, choices=[0, 1], help='fine-tuning backbone')
        # ETC
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: Online-SGD USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)
        
        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='')
        if args.ft_backbone:
            # full fine-tuning
            backbone.feat.requires_grad_(True)
        

        super(OnlineSgd, self).__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        # init task per class
        self.task_per_cls = [0]
    
    def online_before_task(self, task_id):
        self.subset_start = self.task_per_cls[task_id]
        pass

    def online_before_train(self):
        pass
    
    def online_step(self, inputs, labels, not_aug_inputs, idx):
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
        class_to_idx = {label: idx for idx, label in enumerate(self.exposed_classes)}

        x, y = data
        
        for j in range(len(y)):
            y[j] = class_to_idx[y[j].item()]

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
    
    def model_forward(self, x, y):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            logits = self.net(x, return_outputs=True)
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
            loss_dict.update({'total_loss': ce_loss})
                
        return logits, loss_dict
    
    def online_after_task(self, task_id):
        self.task_per_cls.append(len(self.exposed_classes))
        pass

    def online_after_train(self):
        pass

    def get_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]