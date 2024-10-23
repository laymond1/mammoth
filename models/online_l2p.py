"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import torch
import torch.nn.functional as F

from models.utils.online_continual_model import OnlineContinualModel
from utils.args import ArgumentParser
from timm import create_model  # noqa
from models.l2p_utils.l2p_model import L2PModel


class OnlineL2P(OnlineContinualModel):
    """Learning to Prompt (L2P)."""
    NAME = 'online-l2p'
    COMPATIBILITY = ['si-blurry'] # sdp, stream

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
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
        L2P re-defines the backbone model to include the prompt parameters. This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        del backbone
        print("-" * 20)
        print(f"WARNING: L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        # args.lr = args.lr * args.batch_size / 256.0
        backbone = L2PModel(args)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)

    def begin_task(self, dataset):
        self.net.original_model.eval()

        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        
    def online_before_task(self, task_id):
        self.net.original_model.eval()
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        outputs = self.net(inputs, return_outputs=True)
        logits = outputs['logits']

        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        ce_loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            cos_loss = self.args.pull_constraint_coeff * outputs['reduce_sim']
            loss = ce_loss - cos_loss
        else:
            loss = ce_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
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
            # here is the trick to mask out classes of non-current classes
            non_cur_classes_mask = torch.zeros(self.num_classes, device=self.device) - torch.inf
            non_cur_classes_mask[y.unique()] = 0
            # mask out unseen classes and non-current classes
            logits += self.mask + non_cur_classes_mask
            
            ce_loss = self.loss(logits, y)
            if self.args.pull_constraint and 'reduce_sim' in outputs:
                cos_loss = self.args.pull_constraint_coeff * outputs['reduce_sim']
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

    def get_parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]

    # def forward(self, x):
    #     return self.net(x)[:, :self.n_seen_classes]
