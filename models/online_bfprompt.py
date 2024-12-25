"""
BFPrompt: Boundary Free Prompting for Online Continual Learning

Note:
    BFPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import gc
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.online_continual_model import OnlineContinualModel
from models.prompt_utils.model import PromptModel
from models.prompt_utils.prompt import label2prompt
from utils.buffer import Buffer

import wandb


class OnlineBFPrompt(OnlineContinualModel):
    """Boundary Free Prompting (BFPrompt)."""
    NAME = 'online-bfprompt'
    COMPATIBILITY = ['si-blurry', 'periodic-gaussian']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Replay parameters
        add_rehearsal_args(parser)
        # Trick
        parser.add_argument('--train_mask', type=bool, default=True,  help='if using the class mask at training')

        # G-Prompt parameters
        parser.add_argument('--g_prompt_layer_idx', type=int, default=[], nargs="*", help='the layer index of the G-Prompt')
        parser.add_argument('--g_prompt_length', type=int, default=10, help='length of G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[0, 1, 2, 3, 4], nargs="*", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', default=10, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--e_prompt_length', type=int, default=40, help='length of E-Prompt')
        parser.add_argument('--top_k', default=1, type=int, help='top k prompts to use (N in paper)')        
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')
        parser.add_argument('--same_key_value', type=bool, default=True, help='the same key-value across all layers of the E-Prompt')
        parser.add_argument('--gt_key_value', type=int, default=0, choices=[0, 1], help='ground truth key-value')
        parser.add_argument('--prompt_prediction', type=int, default=0, choices=[0, 1], help='prompt prediction with logits')
        
        # SupCon
        parser.add_argument('--use_supcon', default=False, type=bool)
        parser.add_argument('--temperature', default=0.1, type=float, help='temperature for SupCon loss')
        # Random prompt selection
        parser.add_argument('--use_random_selection', default=False, type=bool)

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: BFPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)


        args.e_prompt_pool_size = args.n_tasks
        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        # if num_classes % args.e_prompt_pool_size != 0: # 196 / 10 => 6
            # args.e_prompt_pool_size += 1
        backbone = PromptModel(args, 
                                 num_classes=num_classes,
                                 pretrained=True, prompt_flag='bfprompt',
                                 prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.g_prompt_length, args.temperature])

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
            logits, prompt_loss = self.net(x, y, train=True)
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
            if self.args.pull_constraint_coeff > 0.0:
                cos_loss = self.args.pull_constraint_coeff * prompt_loss
                total_loss = ce_loss + cos_loss
                loss_dict.update({'ce_loss': ce_loss})
                loss_dict.update({'cos_loss': cos_loss})
                loss_dict.update({'total_loss': total_loss})
            else:
                loss_dict.update({'total_loss': ce_loss})
                
        return logits, loss_dict
    
    # TODO: multi-key not considered
    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        top_p_correct, pred_p_correct = 0.0, 0.0
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

                # Define true_p_idx before using it
                true_p_idx = label2prompt(y, cls_per_prompt=self.num_classes//self.net.prompt.e_pool_size, e_pool_size=self.net.prompt.e_pool_size)

                # Prompt prediction with backbone and head classifier 
                if self.args.gt_key_value:
                    logits = self.net(x, y) 
                else:
                    if self.args.prompt_prediction:
                        # prev_logits = self.net.vit_head(feats)[:, :self.num_classes]
                        prev_logits = self.net(x) # top_k
                        prev_logits = prev_logits + self.mask
                        _, p_preds = prev_logits.topk(1, 1, True, True) # select prompt with head
                        # predict logits
                        logits = self.net(x, p_preds)
                        # prompt prediction accuracy
                        pred_p_idx = label2prompt(p_preds, cls_per_prompt=self.num_classes//self.net.prompt.e_pool_size, e_pool_size=self.net.prompt.e_pool_size)
                        pred_p_correct += torch.sum(pred_p_idx == true_p_idx.unsqueeze(1)).item()
                    else:
                        logits = self.net(x)
                logits = logits + self.mask
                # top-k prompt prediction accuracy
                top_p_idx = self.net.prompt.top_k_idx
                top_p_correct += torch.sum(top_p_idx == true_p_idx.unsqueeze(1)).item()

                # count selected prompt when trianing
                num = true_p_idx.view(-1).bincount(minlength=self.net.prompt.e_pool_size)
                self.net.prompt.gt_count += num

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
        avg_top_p_acc = top_p_correct / total_num_data
        avg_pred_p_acc = pred_p_correct / total_num_data

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                     "avg_top_p_acc": avg_top_p_acc, "avg_pred_p_acc": avg_pred_p_acc}
        
        return eval_dict
    
    def online_after_train(self):
        pass
    
    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]