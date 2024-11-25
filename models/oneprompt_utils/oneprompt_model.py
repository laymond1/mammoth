import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from models.oneprompt_utils.vit import VisionTransformer


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p   


class OnePrompt(nn.Module):
    def __init__(self, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}',p)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt length
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
        else:
            p_return = None

        return p_return, 0, x_block


class OnePromptModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, prompt_param=None):
        super(OnePromptModel, self).__init__()

        self.num_classes = num_classes
        self.prompt_param = prompt_param

        # get feature encoder
        self.feat = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                    num_heads=12, ckpt_layer=0, drop_path_rate=0)
        
        if pretrained:
            from timm.models.vision_transformer import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            if 'head.weight' in load_dict:
                del load_dict['head.weight']
                del load_dict['head.bias']
            missing, unexpected = self.feat.load_state_dict(load_dict, strict=False)
            assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        # classifier
        self.last = nn.Linear(768, num_classes)
        # prompt
        self.prompt = OnePrompt(768, prompt_param)
        
    def forward(self, x, train=False, last=False, warmup=False, feat=False, **kwargs):
        if last:
            return self.last(x)

        if self.prompt is not None:
            out, prompt_loss = self.feat(
                x, prompt=self.prompt,
                q=torch.zeros(x.size(0), 768, device=x.device), train=train
            )
            out = out[:, 0, :]
            if warmup:
                prompt_loss = torch.zeros_like(prompt_loss)
                out = out.detach()
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)

        if feat:
            return out
        out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
