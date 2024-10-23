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
import torch.distributed as dist

from timm import create_model  # noqa
from torch.utils.data import DataLoader

from models.utils.online_continual_model import OnlineContinualModel
from models.mvp_utils.mvp_model import MVPModel
from utils.args import ArgumentParser
from utils.mvp_buffer import Memory, MemoryBatchSampler


class MVP(OnlineContinualModel):
    """Learning to Prompt (MVP)."""
    NAME = 'online-mvp'
    COMPATIBILITY = ['si-blurry'] # sdp, stream

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
        # MVP parameters
        parser.add_argument('--use_mask', action='store_true', help='use mask for our method')
        parser.add_argument('--use_contrastiv', action='store_true', help='use contrastive loss for our method')
        parser.add_argument('--use_last_layer', action='store_true', help='use last layer for our method')
        parser.add_argument('--use_afs', action='store_true', help='enable Adaptive Feature Scaling (AFS) in ours')
        parser.add_argument('--use_gsf', action='store_true', help='enable Minor-Class Reinforcement (MCR) in ours')
        
        parser.add_argument('--selection_size', type=int, default=1, help='# candidates to use for ViT_Prompt')
        parser.add_argument('--alpha', type=float, default=0.5, help='# candidates to use for STR hyperparameter')
        parser.add_argument('--gamma', type=float, default=2., help='# candidates to use for STR hyperparameter')
        parser.add_argument('--margin', type=float, default=0.5, help='# candidates to use for STR hyperparameter')
        
        parser.add_argument('--buffer_size', type=int, default=0, help='buffer size for memory replay')

        # Prompt parameters
        parser.add_argument('--prompt_pool', default=True, type=bool,)
        parser.add_argument('--pool_size_l2p', default=10, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--length', default=5, type=int, help='length of prompt (L_p in paper)')
        parser.add_argument('--top_k', default=5, type=int, help='top k prompts to use (N in paper)')
        parser.add_argument('--prompt_key', default=True, type=bool, help='Use learnable prompt key')
        parser.add_argument('--prompt_key_init', default='uniform', type=str, help='initialization type for key\'s prompts')
        parser.add_argument('--use_prompt_mask', default=False, type=bool)
        parser.add_argument('--batchwise_prompt', default=True, type=bool)
        parser.add_argument('--embedding_key', default='cls', type=str)
        parser.add_argument('--predefined_key', default='', type=str)
        parser.add_argument('--pull_constraint', default=True)
        parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

        # Learning rate schedule parameters
        parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
        parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
        parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
        parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
        parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
        parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
        parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
        parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
        parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
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

        # args.lr = args.lr * args.batch_size / 256.0
        backbone = MVPModel(args)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.memory = Memory()
        self.memory_batchsize = args.batch_size - args.temp_batch_size # half of the batch size
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)
    
    def online_before_task(self, task_id):
        pass
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        feature, mask = self.net.forward_features(inputs)
        logits = self.net.forward_head(feature)
        if self.args.use_mask:
            logits = logits * mask
        
        loss_dict = self.loss_fn(feature, mask, labels)
        loss = loss_dict['total_loss']
    
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()
    
    def online_step(self, inputs, labels, idx):
        self.add_new_class(labels)
        
        _loss_dict = dict()
        _acc, _iter = 0.0, 0

        if self.args.buffer_size > 0:
            self.memory.add_new_class(cls_list=self.exposed_classes) 
            self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.args.temp_batch_size * int(self.args.online_iter) * self.world_size)
            self.memory_dataloader   = DataLoader(self.dataset.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=4)
            self.memory_provider     = iter(self.memory_dataloader)

        for _ in range(int(self.args.online_iter)):
            loss_dict, acc = self.online_train([inputs.clone(), labels.clone()])
            _loss_dict = {k: v + _loss_dict.get(k, 0.0) for k, v in loss_dict.items()}
            _acc += acc
            _iter += 1
        
        if self.args.buffer_size > 0:
            self.update_memory(idx, labels)
        del(inputs, labels)
        gc.collect()
        _loss_dict = {k: v / _iter for k, v in _loss_dict.items()}
        return loss_dict, _acc / _iter
    
    def online_train(self, data):
        self.net.train()
        total_loss_dict = dict()
        total_correct, total_num_data = 0.0, 0.0

        x, y = data
        self.labels = torch.cat((self.labels, y), 0)
        
            
        if self.args.buffer_size > 0:
            if len(self.memory) > 0 and self.memory_batchsize > 0:
                memory_images, memory_labels = next(self.memory_provider)
                for i in range(len(memory_labels)):
                    memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
                x = torch.cat([x, memory_images], dim=0)
                y = torch.cat([y, memory_labels], dim=0)

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
        loss_dict = dict()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            feature, mask = self.net.forward_features(x)
            logit = self.net.forward_head(feature)
            if self.args.use_mask:
                logit = logit * mask
            logit = logit + self.mask
            loss_dict = self.loss_fn(feature, mask, y)
            
            return logit, loss_dict
    
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

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.net.backbone.head)
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
        head_w = self.net.backbone.head.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.args.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score
    
    def loss_fn(self, feature, mask, y):
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
            
            total_loss = (1-self.args.alpha)* ce_loss + self.args.alpha * gsf_loss
        else:
            total_loss = ce_loss
            
        total_loss = total_loss.mean() + self.net.get_similarity_loss()
        
        loss_dict.update({'ce_loss': ce_loss.mean()})
        loss_dict.update({'cos_loss': self.net.get_similarity_loss()})
        loss_dict.update({'total_loss': total_loss.mean()})
        
        return loss_dict
    
    def update_memory(self, sample, label):
        # Update memory
        if self.distributed: # TODO: all_gather function is not implemented
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.args.buffer_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,)).item()
                    if j < self.args.buffer_size:
                        idx.append(j)
                    else:
                        idx.append(self.args.buffer_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier() # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.args.buffer_size:
                if index < self.args.buffer_size:
                    self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]], index)
            else:
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])
