"""
MVP: Mask and Visual Prompt Tuning

Note:
    MVP USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import copy
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.online_continual_model import OnlineContinualModel
from models.prompt_utils.model import PromptModel
from utils.buffer import Buffer

import wandb


class MVP(OnlineContinualModel):
    """Learning to Prompt (MVP)."""
    NAME = 'online-mvp'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian'] # sdp, stream

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Replay parameters
        add_rehearsal_args(parser)
        # Trick
        parser.add_argument('--train_mask', type=bool, default=True,  help='if using the class mask at training')

        # MVP parameters
        parser.add_argument('--use_mask', type=bool, default=True, help='use mask for our method')
        parser.add_argument('--use_contrastiv', type=bool, default=True, help='use contrastive loss for our method')
        parser.add_argument('--use_afs', type=bool, default=True, help='enable Adaptive Feature Scaling (AFS) in ours')
        parser.add_argument('--use_gsf', type=bool, default=True, help='enable Minor-Class Reinforcement (MCR) in ours')
        parser.add_argument('--alpha', type=float, default=0.5, help='# candidates to use for STR hyperparameter') # 0.1, 0.3, 0.5, 0.7
        parser.add_argument('--gamma', type=float, default=2., help='# candidates to use for STR hyperparameter') # 0.5, 1.0, 1.5, 2.0, 2.5
        parser.add_argument('--margin', type=float, default=0.5, help='# candidates to use for STR hyperparameter') # 0.1, 0.3, 0.5, 0.7, 0.9

        # G-Prompt parameters
        parser.add_argument('--g_prompt_layer_idx', type=int, default=[0, 1], nargs="+", help='the layer index of the G-Prompt')
        parser.add_argument('--g_prompt_length', type=int, default=10, help='length of G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', default=10, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--e_prompt_length', type=int, default=40, help='length of E-Prompt')
        parser.add_argument('--top_k', default=1, type=int, help='top k prompts to use (N in paper)')        
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        MVP re-defines the backbone model to include the prompt parameters. This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        del backbone
        print("-" * 20)
        print(f"WARNING: MVP USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='mvp',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.g_prompt_length])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # buffer 
        self.buffer = Buffer(self.args.buffer_size)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)
        cols = [f"Prompt_{i}" for i in range(1, args.e_prompt_pool_size+1)]
        cols.insert(0, "N_Samples")
        self.table = wandb.Table(columns=cols)
    
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
            feature, prompt_loss, mask = self.net.forward_features(x, train=True)
            logits = self.net.forward_head(feature)
            if self.args.use_mask:
                logits = logits * mask
            loss_dict = dict()
            # here is the trick to mask out classes of non-current classes
            non_cur_classes_mask = torch.zeros(self.num_classes, device=self.device) - torch.inf
            non_cur_classes_mask[y.unique()] = 0
            # mask out unseen classes and non-current classes
            if self.args.train_mask:
                logits = logits + non_cur_classes_mask
            else:
                logits = logits + self.mask

            loss_dict = self.loss_fn(feature, prompt_loss, mask, y)

            return logits, loss_dict
        
    def online_after_train(self):
        pass
    
    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.net.head)
        head.zero_grad()
        logit = head(feature.detach())
        if self.args.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask
        
        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim
        
        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim
        
        return sample_grad, batch_grad
    
    def _get_ignore(self, sample_grad, batch_grad):
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.net.head.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.args.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score
    
    def loss_fn(self, feature, prompt_loss, mask, y):
        loss_dict = dict()
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.args.use_afs:
            logit = self.net.forward_head(feature)
            logit = self.net.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.net.forward_head(feature)
        if self.args.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        ce_loss = F.nll_loss(log_p, y)
        if self.args.use_gsf:
            gsf_loss = (ign_score ** self.args.gamma) * ce_loss
            loss_dict.update({'gsf_loss': gsf_loss.mean()})
            
            total_loss = (1-self.args.alpha) * ce_loss + self.args.alpha * gsf_loss
        else:
            total_loss = ce_loss
        
        if self.args.pull_constraint_coeff > 0.0:
            cos_loss = self.args.pull_constraint_coeff * prompt_loss
        total_loss = total_loss.mean() + cos_loss
        
        loss_dict.update({'ce_loss': ce_loss.mean()})
        loss_dict.update({'cos_loss': prompt_loss})
        loss_dict.update({'total_loss': total_loss.mean()})
        
        return loss_dict

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]