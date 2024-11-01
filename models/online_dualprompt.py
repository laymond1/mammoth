"""
DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning

Note:
    WARNING: DualPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import logging
import torch
import torch.nn.functional as F
from models.dualprompt_utils.dualprompt_model import DualPromptModel

from models.utils.online_continual_model import OnlineContinualModel, select_optimizer, select_scheduler
from utils.args import ArgumentParser

from datasets import get_dataset


class OnlineDualPrompt(OnlineContinualModel):
    """DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning."""
    NAME = 'online-dualprompt'
    COMPATIBILITY = ['si-blurry']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
        parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

        # Optimizer parameters
        parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')

        # G-Prompt parameters
        parser.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
        parser.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
        parser.add_argument('--g_prompt_layer_idx', default=[0, 1], type=int, nargs="+", help='the layer index of the G-Prompt')
        parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
        parser.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

        # Use prompt pool in L2P to implement E-Prompt
        parser.add_argument('--prompt_pool', default=True, type=bool,)
        parser.add_argument('--size', default=10, type=int,)
        parser.add_argument('--length', default=20, type=int, help='length of E-Prompt')
        parser.add_argument('--top_k', default=1, type=int, )
        parser.add_argument('--initializer', default='uniform', type=str,)
        parser.add_argument('--prompt_key', default=True, type=bool,)
        parser.add_argument('--prompt_key_init', default='uniform', type=str)
        parser.add_argument('--use_prompt_mask', default=False, type=bool)
        parser.add_argument('--mask_first_epoch', default=False, type=bool)
        parser.add_argument('--shared_prompt_pool', default=True, type=bool)
        parser.add_argument('--shared_prompt_key', default=False, type=bool)
        parser.add_argument('--batchwise_prompt', default=True, type=bool)
        parser.add_argument('--embedding_key', default='cls', type=str)
        parser.add_argument('--predefined_key', default='', type=str)
        parser.add_argument('--pull_constraint', default=True)
        parser.add_argument('--pull_constraint_coeff', default=1.0, type=float)
        parser.add_argument('--same_key_value', default=False, type=bool)

        # ViT parameters
        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        logging.info(f"DualPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        backbone = DualPromptModel(args, num_classes=tmp_dataset.N_CLASSES)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)
        
        self.num_cls_per_prompt = self.num_classes // self.args.size

    # def online_before_train(self, task_id):
    #     # 1) task 구분을 num_classes / pool size 로 navive 적용. 
    #     # 2) Si-blurry는 어찌됐든 5 task로 구성되므로 task_id 적용. -> Boundary Free에서는 문제 발생.
    #     # cpp = 10 # 100 // 10 , if exposed_classes are 15 -> task_id 2
    #     # 1) 방식으로 구현, exposed_classes가 한번에 10개 이상 노출되어 task가 2개로 넘어가면, 문제 발생...
    #     # task_id = (len(self.exposed_classes)-1) // self.num_cls_per_prompt
    #     if task_id > 0:
    #         prev_start = (task_id - 1) * self.args.top_k
    #         prev_end = task_id * self.args.top_k

    #         cur_start = prev_end
    #         cur_end = (task_id + 1) * self.args.top_k

    #         if (prev_end > self.args.size) or (cur_end > self.args.size):
    #             pass
    #         else:
    #             cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
    #             prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

    #             with torch.no_grad():
    #                 self.net.model.e_prompt.prompt.grad.zero_()
    #                 self.net.model.e_prompt.prompt[cur_idx] = self.net.model.e_prompt.prompt[prev_idx]
    #                 self.opt.param_groups[0]['params'] = self.net.model.parameters()

    def online_before_train(self):
        pass

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
        class_to_idx = {label: idx for idx, label in enumerate(self.exposed_classes)}

        x, y = data
        self.labels = torch.cat((self.labels, y), 0)
        
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
            logits = self.net(x)
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
            if self.args.pull_constraint > 0.0:
                cos_loss = self.args.pull_constraint_coeff * self.net.get_similarity_loss()
                total_loss = ce_loss - cos_loss
                loss_dict.update({'ce_loss': ce_loss})
                loss_dict.update({'cos_loss': cos_loss})
                loss_dict.update({'total_loss': total_loss})
            else:
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

                logits = self.net(x)
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
        return [p for p in self.net.parameters() if p.requires_grad]
    