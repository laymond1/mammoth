import torch
from torch import Tensor
import torch.nn as nn
from functools import partial
from typing import Any, Callable, List, Optional, Sequence
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig, \
                                           _mobilenet_v3_conf

from backbone import MammothBackbone, register_backbone


class MobileNetV3(MammothBackbone):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features_layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, returnt='out') -> Tensor:
        x = self.features_layers(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        if returnt == 'features':
            return feature

        out = self.classifier(feature)
        
        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        else:
            raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both'] but got {}".format(returnt))

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


@register_backbone("mobilenet_v3_small")
def mobilenet_v3(num_classes: int) -> MobileNetV3:
    """MobileNetV3 model architecture from the
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.
    
    Args:
        num_classes (int): Number of classes for the classifier
        **kwargs: Additional keyword arguments for the model
    """
    weights, progress = None, False
    kwargs = {'num_classes': num_classes}
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


@register_backbone("mobilenet_v3_small_pt")
def mobilenet_v3(num_classes: int) -> MobileNetV3:
    """MobileNetV3 model architecture from the
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.
    
    Args:
        num_classes (int): Number of classes for the classifier
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

    weights, progress = None, False
    kwargs = {'num_classes': num_classes}
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    net = _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)

    pretrain_weights = MobileNet_V3_Small_Weights.DEFAULT
    st = pretrain_weights.get_state_dict(progress=True, check_hash=True)
    # ensure no classifier weights are loaded
    for k in list(st.keys()):
        if k.startswith('features'):
            st[k.replace('features.', 'features_layers.')] = st.pop(k)
        if k.startswith('classifier'):
            del st[k]
    missing, unexp = net.load_state_dict(st, strict=False)
    assert len([k for k in missing if not k.startswith('classifier')]) == 0, missing
    assert len(unexp) == 0, unexp
    
    return net


@register_backbone("mobilenet_v3_large")
def mobilenet_v3(num_classes: int) -> MobileNetV3:
    """MobileNetV3 model architecture from the
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.
    
    Args:
        num_classes (int): Number of classes for the classifier
        **kwargs: Additional keyword arguments for the model
    """
    weights, progress = None, False
    kwargs = {'num_classes': num_classes}
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


@register_backbone("mobilenet_v3_large_pt")
def mobilenet_v3(num_classes: int) -> MobileNetV3:
    """MobileNetV3 model architecture from the
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.
    
    Args:
        num_classes (int): Number of classes for the classifier
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

    weights, progress = None, False
    kwargs = {'num_classes': num_classes}
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    net = _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)

    pretrain_weights = MobileNet_V3_Large_Weights.DEFAULT
    st = pretrain_weights.get_state_dict(progress=True, check_hash=True)
    # ensure no classifier weights are loaded
    for k in list(st.keys()):
        if k.startswith('features'):
            st[k.replace('features.', 'features_layers.')] = st.pop(k)
        if k.startswith('classifier'):
            del st[k]
    missing, unexp = net.load_state_dict(st, strict=False)
    assert len([k for k in missing if not k.startswith('classifier')]) == 0, missing
    assert len(unexp) == 0, unexp
    
    return net