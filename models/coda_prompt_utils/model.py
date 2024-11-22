import torch
import torch.nn as nn
from backbone.vit import create_vision_transformer
from models.coda_prompt_utils import gram_schmidt
from models.coda_prompt_utils.vit import VisionTransformer


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, prompt_param, key_dim=768):
        super().__init__()
        self.emb_d = emb_d
        self.key_d = key_dim
        self._init_smart(emb_d, prompt_param)

        pt = self.e_pool_size # no task boundary

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = gram_schmidt(p, start_c=0, end_c=pt)
            k = gram_schmidt(k, start_c=0, end_c=pt)
            a = gram_schmidt(a, start_c=0, end_c=pt)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        # for no task boundary, we removed the prompt selection process based on task id
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).to(t.device))**2).mean()


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class CODAPromptModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, prompt_param=None):
        super().__init__()

        self.task_id = None

        # get feature encoder
        vit_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                      num_heads=12, drop_path_rate=0)

        if pretrained:
            load_dict = create_vision_transformer('vit_base_patch16_224', base_class=VisionTransformer, pretrained=True, num_classes=0).state_dict()
            if 'head.weight' in load_dict:
                del load_dict['head.weight']
                del load_dict['head.bias']
            missing, unexpected = vit_model.load_state_dict(load_dict, strict=False)
            assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        # classifier
        self.last = nn.Linear(768, num_classes)

        self.prompt = CodaPrompt(768, prompt_param)

        # feature encoder changes if transformer vs resnet
        self.feat = vit_model

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False, **kwargs):

        if self.prompt is not None:
            with torch.no_grad():
                query, _ = self.feat(x)
                query = query[:, 0, :]
            out, prompt_loss = self.feat(x, prompt=self.prompt, query=query, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
