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
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian'] # sdp, stream

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
        parser.add_argument('--pull_constraint_coeff', default=0.5, type=float)

        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

        # Learning rate schedule parameters
        parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant")')
        parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
        parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
        parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
        parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
        parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
        parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10)')
        parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
        parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        # Trick parameters
        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
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

        backbone = L2PModel(args)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # set optimizer and scheduler
        self.reset_opt()
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.labels = torch.empty(0)
        
    def online_before_task(self, task_id):
        self.net.original_model.eval()
        
    def online_before_train(self):
        self.net.original_model.eval()
    
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
            outputs = self.net(x, return_outputs=True)
            logits = outputs['logits']
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
            if self.args.pull_constraint and 'reduce_sim' in outputs:
                cos_loss = self.args.pull_constraint_coeff * outputs['reduce_sim']
                total_loss = ce_loss - cos_loss
                loss_dict.update({'ce_loss': ce_loss})
                loss_dict.update({'cos_loss': cos_loss})
                loss_dict.update({'total_loss': total_loss})
            else:
                loss_dict.update({'total_loss': ce_loss})
                
        return logits, loss_dict
    
    def online_after_task(self, task_id):
        pass
    
    def online_after_train(self):
        pass

    def get_parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]