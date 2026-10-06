import torch
from torch import nn
from tqdm import tqdm

from models.tak_utils.hooks import hook_forward_batch
from models.tak_utils.hooks import hook_forward_batch_nosequence
from models.tak_utils.hooks import hook_forward_store_inputs

from models.tak_utils.hooks import hook_backward
from models.tak_utils.hooks import hook_backward_nosequence
from models.tak_utils.hooks import hook_backward_layer_norm
from models.tak_utils.hooks import hook_backward_cls_token

from models.tak_utils.utils import set_requires_grad_to


def get_split(dataset):
    return dataset.train_loader


def register_hooks(name, module, forward=True, backward=True,
                   forward_hooks_dict=None, bacward_hooks_dict=None):

    module.name = name

    if forward:
        assert forward_hooks_dict is not None
        if 'lin_proj' in name:
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_nosequence'])
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward'])
        elif isinstance(module, nn.LayerNorm):
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_layer_norm'])
        elif 'cls_token' in name:
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_layer_norm'])

    if backward:
        assert bacward_hooks_dict is not None
        if 'lin_proj' in name:
            module.backward_handle = module.register_full_backward_hook(bacward_hooks_dict['hook_backward_nosequence'])
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.backward_handle = module.register_full_backward_hook(bacward_hooks_dict['hook_backward'])
        elif isinstance(module, nn.LayerNorm):
            module.backward_handle = module.register_full_backward_hook(bacward_hooks_dict['hook_backward_layer_norm'])
        elif 'cls_token' in name:
            module.backward_handle = module.register_full_backward_hook(bacward_hooks_dict['hook_backward_cls_token'])


class KFACComputer(nn.Module):

    def __init__(self, device: torch.device, debug_mode,
                 train_percent: float = 1.0, num_samples_expectation: int = 2, fp_precision: str = 'fp64'):

        super().__init__()

        if isinstance(train_percent, float):
            assert 0 < train_percent <= 1.0
        elif isinstance(train_percent, int):
            assert train_percent >= 1

        self.device = device
        self.debug_mode = debug_mode
        self.train_percent = train_percent
        self.num_samples_expectation = num_samples_expectation
        self.fp_precision = fp_precision

    def to_be_fishered(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.Linear) \
                and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear) \
                and not isinstance(module, nn.MultiheadAttention):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False

    def to_be_fishered_layer_norm(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.LayerNorm):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False

    def compute(self, net, head, delta_w_names, dataset, use_head=False):

        all_param_finetuned = list(delta_w_names)
        if isinstance(self.train_percent, float):
            num_of_batches = int(self.train_percent * len(dataset.train_loader))
        elif isinstance(self.train_percent, int):
            num_of_batches = self.train_percent
        else:
            raise ValueError("train_percent must be float or int")

        forward_hooks_dict = {
            'hook_forward': hook_forward_batch,
            'hook_forward_nosequence': hook_forward_batch_nosequence,
        }

        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                module.fp_precision = self.fp_precision
                register_hooks(name, module, forward=True, backward=False,
                               forward_hooks_dict=forward_hooks_dict)

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        num_of_examples_aaT = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(get_split(dataset)),
                                total=len(get_split(dataset)),
                                desc='aaT computation'):

                if self.debug_mode and i > 3:
                    break

                if i >= num_of_batches:
                    break

                x = data[0].to(self.device)
                num_of_examples_aaT += x.shape[0]
                _ = net.visual_encoder(x)

        aaT = {}

        def collect_aaT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                aaT[f"{name}.weight"] = getattr(module, "gram_input")

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_aaT(name, module)

        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.gram_input = None
                module.gram_input_c = None
                del module.gram_input
                del module.gram_input_c
                del module.fp_precision

        ##################

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict_layer_norm = {
            'hook_forward_layer_norm': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward,
            'hook_backward_nosequence': hook_backward_nosequence,
        }

        backward_hooks_dict_layer_norm = {
            'hook_backward_layer_norm': hook_backward_layer_norm,
        }

        backward_hooks_dict_cls_token = {
            'hook_backward_cls_token': hook_backward_cls_token,
        }

        for name, module in net.visual_encoder.named_modules():
            module.fp_precision = self.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=False, backward=True,
                               bacward_hooks_dict=backward_hooks_dict)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_layer_norm,
                               forward_hooks_dict=forward_hooks_dict_layer_norm)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                register_hooks(name, module, forward=False, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_cls_token)

        num_of_examples_ggT = 0

        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='ggT computation'):

            if self.debug_mode and i > 3:
                break

            if i >= num_of_batches:
                break

            x = data[0].to(self.device)

            num_of_examples_ggT += x.shape[0]

            features = net.visual_encoder(x * fake_param)
            features = features / features.norm(dim=-1, keepdim=True)

            if use_head:
                features = head(features)

            if self.num_samples_expectation > 0:
                for s in range(self.num_samples_expectation):
                    (features * torch.randn_like(features)).sum().backward(
                        retain_graph=s < self.num_samples_expectation - 1)
            else:
                features = features.sum(0)
                for cnt_class, feat in enumerate(features):
                    fake_optim.zero_grad()
                    feat.backward(retain_graph=cnt_class < features.shape[0] - 1)

        fake_optim.zero_grad()

        ggT = {}
        ffT = {}

        def collect_ggT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                ggT[f"{name}.weight"] = getattr(module, "gram_grad")

        def collect_ffT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                ffT[f"{name}.weight"] = getattr(module, "gram_grad_weight")
            if f"{name}.bias" in all_param_finetuned:
                ffT[f"{name}.bias"] = getattr(module, "gram_grad_bias")

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_ggT(name, module)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                ffT[f'{name}.class_embedding'] = getattr(module, "gram_grad")

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            del module.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.backward_handle.remove()
                module.gram_grad = None
                module.gram_grad_c = None
                del module.gram_grad
                del module.gram_grad_c
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.inputs = None
                module.gram_grad_weight = None
                module.gram_grad_weight_c = None
                module.gram_grad_bias = None
                module.gram_grad_bias_c = None
                del module.inputs
                del module.gram_grad_weight
                del module.gram_grad_weight_c
                del module.gram_grad_bias
                del module.gram_grad_bias_c
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                module.backward_handle.remove()
                module.inputs = None
                module.gram_grad = None
                module.gram_grad_c = None
                del module.gram_grad
                del module.gram_grad_c

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return ggT, aaT, ffT, num_of_examples_ggT, num_of_examples_aaT