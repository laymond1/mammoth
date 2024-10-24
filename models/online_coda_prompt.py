"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import logging
import torch
import torch.nn.functional as F

from models.coda_prompt_utils.model import Model
from models.utils.online_continual_model import OnlineContinualModel
from utils.args import *
from utils.schedulers import CosineSchedule

from datasets import get_dataset


class CodaPrompt(OnlineContinualModel):
    """Continual Learning via CODA-Prompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'online-coda-prompt'
    COMPATIBILITY = ['online-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--mu', type=float, default=0.1, help='weight of prompt loss')
        parser.add_argument('--pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--prompt_len', type=int, default=8, help='prompt length')
        parser.add_argument('--virtual_bs_iterations', '--virtual_bs_n', dest='virtual_bs_iterations',
                            type=int, default=1, help="virtual batch size iterations")
        # Optimizer parameters
        parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
        # Trick parameters
        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        logging.info(f"CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        if args.lr_scheduler is not None:
            logging.info("CODA-Prompt uses a custom scheduler: cosine. Ignoring --lr_scheduler.")

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        n_tasks = args.n_tasks
        num_classes =tmp_dataset.N_CLASSES
        
        backbone = Model(num_classes=num_classes,
                         pt=True, 
                         prompt_param=[n_tasks, [args.pool_size, args.prompt_len, 0]])
        
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.net.task_id = 0
        self.opt = self.get_optimizer()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)

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

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self.dataset.get_offsets(self.current_task)

        if self.current_task != 0:
            self.net.task_id = self.current_task
            self.net.prompt.process_task_count()
            self.opt = self.get_optimizer()

        self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
        
    def online_before_task(self, task_id):
        if task_id != 0:
            self.net.task_id = task_id
            self.net.prompt.process_task_count()
            self.opt = self.get_optimizer()
            
        self.scheduler = CosineSchedule(self.opt, K=self.args.online_iter) # TODO: check if this is correct

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        labels = labels.long()
        self.opt.zero_grad()
        logits, loss_prompt = self.net(inputs, train=True)
        loss_prompt = loss_prompt.sum()
        logits = logits[:, :self.offset_2]
        logits[:, :self.offset_1] = -float('inf')
        loss_ce = self.loss(logits, labels)
        loss = loss_ce + self.args.mu * loss_prompt
        if self.task_iteration == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        (loss / float(self.args.virtual_bs_iterations)).backward()
        if self.task_iteration > 0 and self.task_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

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
            total_loss = ce_loss + self.args.mu * prompt_loss
            
            loss_dict.update({'ce_loss': ce_loss})
            loss_dict.update({'prompt_loss': prompt_loss})
            loss_dict.update({'total_loss': total_loss})
            
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

    # def forward(self, x):
    #     return self.net(x)[:, :self.offset_2]
