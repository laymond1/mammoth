import munch
import yaml
import torch
import torch.nn as nn
from models.superprompt_utils.elasticvit.checkpoint import load_checkpoint
from models.superprompt_utils.elasticvit.build_model import build_supernet
from models.superprompt_utils.prompt import CodaPrompt


class PromptModel(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False, prompt_flag=False, prompt_param=None):
        super().__init__()
        self.args = args
        # select prompt method
        self.num_classes = num_classes
        self.prompt_flag = prompt_flag
        # select FLOPs ranges
        # self.flops_ranges = self.args.flops_ranges
        with open(args.config_file) as yaml_file:
            config = yaml.safe_load(yaml_file)
        self.config = munch.munchify(config)
        
        if pretrained:
            pretrained_model, _, _ = build_supernet(self.config)
            pretrained_model, start_epoch, extras = load_checkpoint(
                pretrained_model, 
                self.config.resume.path, 
                strict=True, 
                lean=False
            )
            self.feat = pretrained_model
            self.feat.requires_grad_(False)
            
            # Get the feature dimension from pre_head_1
            feature_dim = self.feat.pre_head_1.out_features
            self.head = nn.Linear(feature_dim, num_classes)
            
        args.e_prompt_super_dim = {}
        for e in args.e_prompt_layer_idx:
            args.e_prompt_super_dim[e] = self.feat.feature_exactor[e].layers[0].v_super_dim
        self.prompt = CodaPrompt(args, feature_dim)
            
    def forward_features(self, x, prompt=None, q=None, train=False):
        """Extract features up to pre_head_1 (before the built-in classifier)"""
        if x.size(-1) != self.feat.input_res:
            x = torch.nn.functional.interpolate(
                x, size=self.feat.input_res, mode='bicubic')

        first_conv_weights = self.feat.first_conv.weight[:
                                                        self.feat.sampled_first_conv_channels, :3]
        x = self.feat.first_conv._conv_forward(x, first_conv_weights, bias=None)
        x = self.feat.first_conv_act(self.feat.first_conv_bn(x))
        x = self.feat.conv_stem(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        
        for idx, module in enumerate(self.feat.feature_exactor):
            if idx > 1:
                if self.feat.stage[idx-1] == 'C' and self.feat.stage[idx] == 'T':
                    x = x.flatten(2).transpose(1, 2)

            if self.feat.stage[idx] == 'T':   
                if prompt is not None:
                    # activated_qk_dim: sampled_qkdim * sampled_num_heads
                    # activated_v_dim: sampled_vdim * sampled_num_heads
                    activated_qk_dim, activated_v_dim = module.layers[0].qkv.activated_out_dim
                    
                    if train:
                        p_list, loss, x = prompt.forward(q, idx, x, activated_qk_dim, activated_v_dim, train=True)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, idx, x, activated_qk_dim, activated_v_dim, train=False)
                else:
                    p_list = None
                x = module(x, prompt=p_list)
            else:
                x = module(x)

        # 1 x 4 x 160
        self.feat._set_mbv3_head()

        x = self.feat.pre_head_act(self.feat.pre_head_norm_0(
            self.feat.pre_head_0(x))).mean(dim=1).squeeze(1)
        x = self.feat.pre_head_act(self.feat.pre_head_1(x))
        if self.feat.head_dropout_prob > 0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.feat.head_dropout_prob)
        # 1 x 1984
        return x, prompt_loss
            
    def forward(self, x, train=False, last=False, feat=False):
        if last:
            return self.head(x)
        
        with torch.no_grad():
            q, _ = self.forward_features(x)
        
        out, prompt_loss = self.forward_features(x, q=q, prompt=self.prompt, train=train)
        
        if feat:
            return out
        out = self.head(out)
        
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out