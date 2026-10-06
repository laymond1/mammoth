import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F

from models.tak_utils.hooks import hook_forward_store_inputs

from models.tak_utils.utils import set_requires_grad_to


def get_split(dataset):
    return dataset.train_loader


@torch.no_grad()
def hook_backward_diag(module, _, grad_output):
    if module.fp_precision == 'fp32':
        grad_out = grad_output[0].float()
        inputs = module.inputs.float()
    elif module.fp_precision == 'fp64':
        grad_out = grad_output[0].double()
        inputs = module.inputs.double()
    else:
        raise NotImplementedError

    if len(grad_out.shape) > 2:
        if 'attn.proj' in module.name or 'attn.qkv' in module.name:
            B, R, C = grad_out.shape
        else:
            R, B, C = grad_out.shape
            grad_out = grad_out.permute(1, 0, 2)
            inputs = inputs.permute(1, 0, 2)
        grad_weight = torch.einsum('blo,bli->boi', grad_out, inputs)
    else:
        grad_weight = torch.einsum('bo,bi->boi', grad_out, inputs)

    grad_bias = None

    if hasattr(module, "bias") and module.compute_bias:
        if len(grad_out.shape) > 2:
            grad_bias = grad_out.sum(1)
        else:
            assert False

    if grad_bias is not None:
        grad_weight = torch.cat((grad_weight, grad_bias.unsqueeze(2)), dim=2)

    grad_weight = grad_weight.pow(2).sum(0)

    if not hasattr(module, "grad_weight"):
        module.grad_weight = torch.zeros_like(grad_weight)

    module.grad_weight.add_(grad_weight)


@torch.no_grad()
def hook_backward_layer_norm_diag(module, _, grad_output):
    if module.fp_precision == 'fp32':
        grad_out = grad_output[0].float()
        inputs = module.inputs.float()
        normalized = F.layer_norm(inputs, module.normalized_shape).float()
    elif module.fp_precision == 'fp64':
        grad_out = grad_output[0].double()
        inputs = module.inputs.double()
        normalized = F.layer_norm(inputs, module.normalized_shape).double()
    else:
        raise NotImplementedError

    grad_weight = grad_out * normalized  # un-batched grad wrt weights

    if len(grad_out.shape) > 2:
        if "ln_pre" in module.name:
            grad_weight = grad_weight.sum(1)
        else:
            grad_weight = grad_weight.sum(0)

    if hasattr(module, "bias") and module.compute_bias:

        grad_bias = grad_out

        if len(grad_out.shape) > 2:
            if "ln_pre" in module.name:
                grad_bias = grad_bias.sum(1)
            else:
                grad_bias = grad_bias.sum(0)

        grad_weight = torch.cat((grad_weight.unsqueeze(2), grad_bias.unsqueeze(2)), dim=2)

    grad_weight = grad_weight.pow(2).sum(0)

    if not hasattr(module, "grad_weight"):
        module.grad_weight = torch.zeros_like(grad_weight)

    module.grad_weight.add_(grad_weight)


@torch.no_grad()
def hook_backward_cls_token_diag(module, _, grad_output):
    if module.fp_precision == 'fp32':
        grad_out = grad_output[0].float()
    elif module.fp_precision == 'fp64':
        grad_out = grad_output[0].double()
    else:
        raise NotImplementedError

    grad_weight = grad_out[:, 0].pow(2).sum(0)
    if not hasattr(module, "grad_weight"):
        module.grad_weight = torch.zeros_like(grad_weight)
    module.grad_weight.add_(grad_weight)


def register_hooks(name, module, forward=True, backward=True,
                   forward_hooks_dict=None, bacward_hooks_dict=None):
    module.name = name

    if forward:
        assert forward_hooks_dict is not None
        if 'lin_proj' in name:
            module.forward_handle = module.register_forward_hook(
                forward_hooks_dict['hook_forward_nosequence']) 
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward']) 
        elif isinstance(module, nn.LayerNorm):
            module.forward_handle = module.register_forward_hook(
                forward_hooks_dict['hook_forward_layer_norm']) 
        elif 'cls_token' in name:
            module.forward_handle = module.register_forward_hook(
                forward_hooks_dict['hook_forward_layer_norm']) 

    if backward:
        assert bacward_hooks_dict is not None
        if 'lin_proj' in name:
            module.backward_handle = module.register_full_backward_hook(
                bacward_hooks_dict['hook_backward_nosequence']) 
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.backward_handle = module.register_full_backward_hook(
                bacward_hooks_dict['hook_backward']) 
        elif isinstance(module, nn.LayerNorm):
            module.backward_handle = module.register_full_backward_hook(
                bacward_hooks_dict['hook_backward_layer_norm']) 
        elif 'cls_token' in name:
            module.backward_handle = module.register_full_backward_hook(
                bacward_hooks_dict['hook_backward_cls_token']) 


class DiagComputer(nn.Module):

    def __init__(self, device: torch.device, debug_mode,
                 train_percent: float = 1.0, num_samples_expectation: int = 0, fp_precision: str = 'fp64'):

        super().__init__()

        assert 0 < train_percent <= 1.0

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
        num_of_batches = int(self.train_percent * len(dataset.train_loader))

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict = {
            'hook_forward': hook_forward_store_inputs,
            'hook_forward_nosequence': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward_diag,
            'hook_backward_nosequence': hook_backward_diag,
        }

        forward_hooks_dict_layer_norm = {
            'hook_forward_layer_norm': hook_forward_store_inputs,
        }

        backward_hooks_dict_layer_norm = {
            'hook_backward_layer_norm': hook_backward_layer_norm_diag,
        }

        backward_hooks_dict_cls_token = {
            'hook_backward_cls_token': hook_backward_cls_token_diag,
        }

        for name, module in net.visual_encoder.named_modules():
            module.fp_precision = self.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               forward_hooks_dict=forward_hooks_dict,
                               bacward_hooks_dict=backward_hooks_dict)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_layer_norm,
                               forward_hooks_dict=forward_hooks_dict_layer_norm)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                register_hooks(name, module, forward=False, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_cls_token)

        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='Fisher diagonal computation'):

            if self.debug_mode and i > 1:
                break

            if i > num_of_batches:
                break

            x = data[0].to(self.device)

            num_of_examples += x.shape[0]

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
                    # fake_optim.zero_grad()
                    feat.backward(retain_graph=cnt_class < features.shape[0] - 1)

        fake_optim.zero_grad()

        ffT = {}

        def collect_ffT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                ffT[f"{name}.weight"] = getattr(module, "grad_weight")

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                ffT[f'{name}.class_embedding'] = getattr(module, "grad_weight")

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            del module.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.grad_weight = None
                module.inputs = None
                del module.inputs
                del module.grad_weight
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.inputs = None
                module.grad_weight = None
                del module.inputs
                del module.grad_weight
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                module.backward_handle.remove()
                module.grad_weight = None
                del module.grad_weight

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return ffT, num_of_examples


class LossDiagComputer(DiagComputer):

    def __init__(self, device: torch.device, debug_mode,
                 train_percent: float = 1.0, fp_precision: str = 'fp64'):
        super().__init__(device, debug_mode, train_percent, fp_precision=fp_precision)

    def compute(self, net, head, delta_w_names, dataset, use_head=False):

        assert use_head is True and head is not None

        all_param_finetuned = list(delta_w_names)
        num_of_batches = int(self.train_percent * len(dataset.train_loader))

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict = {
            'hook_forward': hook_forward_store_inputs,
            'hook_forward_nosequence': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward_diag,
            'hook_backward_nosequence': hook_backward_diag,
        }

        forward_hooks_dict_layer_norm = {
            'hook_forward_layer_norm': hook_forward_store_inputs,
        }

        backward_hooks_dict_layer_norm = {
            'hook_backward_layer_norm': hook_backward_layer_norm_diag,
        }

        backward_hooks_dict_cls_token = {
            'hook_backward_cls_token': hook_backward_cls_token_diag,
        }

        for name, module in net.visual_encoder.named_modules():
            module.fp_precision = self.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               forward_hooks_dict=forward_hooks_dict,
                               bacward_hooks_dict=backward_hooks_dict)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_layer_norm,
                               forward_hooks_dict=forward_hooks_dict_layer_norm)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                register_hooks(name, module, forward=False, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_cls_token)

        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='Fisher diagonal computation'):

            if self.debug_mode and i > 1:
                break

            if i > num_of_batches:
                break

            x = data[0].to(self.device)

            num_of_examples += x.shape[0]

            features = net.visual_encoder(x * fake_param)
            features = features / features.norm(dim=-1, keepdim=True)

            if use_head:
                features = head(features)

            probs = torch.softmax(features, dim=1)
            detached_probs = probs.detach()
            log_probs = torch.log(probs)
            fisher_sqrt = (detached_probs.sqrt() * log_probs).sum(0)

            for cnt_class, fish in enumerate(fisher_sqrt):
                fish.backward(
                    retain_graph=True if (cnt_class < fisher_sqrt.shape[0] - 1) else False
                )

        fake_optim.zero_grad()

        ffT = {}

        def collect_ffT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                ffT[f"{name}.weight"] = getattr(module, "grad_weight")

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                ffT[f'{name}.class_embedding'] = getattr(module, "grad_weight")

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            del module.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.grad_weight = None
                module.inputs = None
                del module.inputs
                del module.grad_weight
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.inputs = None
                module.grad_weight = None
                del module.inputs
                del module.grad_weight
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                module.backward_handle.remove()
                module.grad_weight = None
                del module.grad_weight

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return ffT, num_of_examples


class LossDiagComputerSampling(DiagComputer):

    def __init__(self, device: torch.device, debug_mode,
                 train_percent: float = 1.0, fp_precision: str = 'fp64'):
        super().__init__(device, debug_mode, train_percent, fp_precision=fp_precision)

    def compute(self, net, head, delta_w_names, dataset, use_head=False):

        assert use_head is True and head is not None

        all_param_finetuned = list(delta_w_names)
        num_of_batches = int(self.train_percent * len(dataset.train_loader))

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict = {
            'hook_forward': hook_forward_store_inputs,
            'hook_forward_nosequence': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward_diag,
            'hook_backward_nosequence': hook_backward_diag,
        }

        forward_hooks_dict_layer_norm = {
            'hook_forward_layer_norm': hook_forward_store_inputs,
        }

        backward_hooks_dict_layer_norm = {
            'hook_backward_layer_norm': hook_backward_layer_norm_diag,
        }

        backward_hooks_dict_cls_token = {
            'hook_backward_cls_token': hook_backward_cls_token_diag,
        }

        for name, module in net.visual_encoder.named_modules():
            module.fp_precision = self.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               forward_hooks_dict=forward_hooks_dict,
                               bacward_hooks_dict=backward_hooks_dict)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                register_hooks(name, module, forward=True, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_layer_norm,
                               forward_hooks_dict=forward_hooks_dict_layer_norm)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                register_hooks(name, module, forward=False, backward=True,
                               bacward_hooks_dict=backward_hooks_dict_cls_token)

        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='Fisher diagonal computation'):

            if self.debug_mode and i > 1:
                break

            if i > num_of_batches:
                break

            x = data[0].to(self.device)

            num_of_examples += x.shape[0]

            features = net.visual_encoder(x * fake_param)
            features = features / features.norm(dim=-1, keepdim=True)

            if use_head:
                features = head(features)

            dist = torch.distributions.Categorical(logits=features)
            y_sample = dist.sample()

            logp_y = features.gather(1, y_sample.unsqueeze(1)).sum(0)

            for cnt_class, fish in enumerate(logp_y):
                fish.backward(
                    retain_graph=True if (cnt_class < logp_y.shape[0] - 1) else False
                )

        fake_optim.zero_grad()

        ffT = {}

        def collect_ffT(name, module):
            if f"{name}.weight" in all_param_finetuned:
                ffT[f"{name}.weight"] = getattr(module, "grad_weight")

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                collect_ffT(name, module)
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                ffT[f'{name}.class_embedding'] = getattr(module, "grad_weight")

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            del module.fp_precision
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.grad_weight = None
                module.inputs = None
                del module.inputs
                del module.grad_weight
            if self.to_be_fishered_layer_norm(name, module, all_param_finetuned):
                del module.compute_bias
                module.forward_handle.remove()
                module.backward_handle.remove()
                module.inputs = None
                module.grad_weight = None
                del module.inputs
                del module.grad_weight
            if 'cls_token' in name and 'cls_token_layer.class_embedding' in all_param_finetuned:
                module.backward_handle.remove()
                module.grad_weight = None
                del module.grad_weight

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return ffT, num_of_examples

