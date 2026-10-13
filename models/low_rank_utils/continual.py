"""Native Mammoth hooks for the pinned CL-LoRA repository's four methods."""

import copy
from abc import abstractmethod
from argparse import ArgumentParser

import torch
from torch import nn
from torchvision import transforms

from datasets.utils.continual_dataset import MammothDatasetWrapper
from models.utils.continual_model import ContinualModel
from utils.conf import create_seeded_dataloader


class TensorList(nn.Module):
    """Variable-size algorithm tensors that remain in ordinary state_dicts."""

    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __getitem__(self, index):
        return self._buffers[str(index)]

    def __setitem__(self, index, value):
        self._buffers[str(index)] = value

    def append(self, value):
        self.register_buffer(str(len(self)), value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = {key[len(prefix):]: value for key, value in state_dict.items()
                 if key.startswith(prefix) and key[len(prefix):].isdigit()}
        self._buffers.clear()
        for key in sorted(saved, key=int):
            self.register_buffer(key, torch.empty_like(saved[key]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def source_transforms(dataset):
    """Install the original image pipeline on this dataset instance only."""
    name = getattr(dataset, 'NAME', '')
    if name in ('seq-cifar100-224', 'seq-cifar10-224'):
        training = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        testing = [transforms.Resize(224)]
    elif name in ('seq-imagenet-r', 'seq-imagenet-a', 'seq-cub200'):
        training = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        interpolation = transforms.InterpolationMode.BILINEAR
        if name in ('seq-imagenet-a', 'seq-cub200'):
            training[0] = transforms.RandomResizedCrop(224, scale=(0.05, 1.0))
            interpolation = transforms.InterpolationMode.BICUBIC
        testing = [transforms.Resize(256, interpolation=interpolation), transforms.CenterCrop(224)]
    else:
        return
    identity = transforms.Normalize((0., 0., 0.), (1., 1., 1.))
    dataset.TRANSFORM = transforms.Compose(training + [transforms.ToTensor(), identity])
    dataset.TEST_TRANSFORM = transforms.Compose(testing + [transforms.ToTensor(), identity])
    dataset.get_normalization_transform = lambda: identity
    dataset.get_denormalization_transform = lambda: identity


class SourceLoader:
    """Translate native batches to the source's (index, image, label) layout."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield None, batch[0], batch[1]


class LowRankContinualModel(ContinualModel):
    """Shared interfaces; each METHOD keeps its own source algorithm state."""

    COMPATIBILITY = ['class-il', 'task-il']

    @property
    @abstractmethod
    def METHOD(self):
        """Subclasses select one source method with their METHOD constant."""
        raise NotImplementedError

    @classmethod
    def get_parser(cls, parser: ArgumentParser) -> ArgumentParser:
        rates = {'ewclora': (0.0005, 0.005), 'sdlora': (0.008, 0.008),
                 'inflora': (0.0005, 0.0005), 'cllora': (0.03, 0.03)}
        lr, fc_lr = rates[cls.METHOD]
        cl = cls.METHOD == 'cllora'
        parser.set_defaults(backbone='low-rank-vit', low_rank_method=cls.METHOD,
                            lr=lr, optimizer='sgd' if cl else 'adam',
                            optim_wd=0.0001 if cl else 0., optim_mom=0.9,
                            optim_nesterov=0, n_epochs=30 if cl else 20,
                            batch_size=64, seed=1993, lr_scheduler='cosine',
                            scheduler_mode='epoch', enable_other_metrics=False)
        parser.add_argument('--lora_vit_size', choices=['base', 'small', 'tiny'], default='base')
        parser.add_argument('--lora_checkpoint', type=str, default=None)
        parser.add_argument('--lora_rank', type=int, default=10)
        parser.add_argument('--lora_fc_lr', type=float, default=fc_lr)
        parser.add_argument('--lora_gamma', type=float, default=1.)
        parser.add_argument('--lora_ewc_lambda', type=float, default=1e7)
        parser.add_argument('--lora_ewc_merge', choices=['factors', 'product'], default='factors',
                            help='EWC-LoRA merge after a task: "factors" adds the A and B factors separately, as the '
                                 'released code does; "product" adds the product BA to the weight, as the paper states')
        parser.add_argument('--lora_lamb', type=float, default=0.95)
        parser.add_argument('--lora_lame', type=float, default=1.)
        parser.add_argument('--lora_kd_ratio', type=float, default=5.)
        parser.add_argument('--lora_temperature', type=float, default=2.)
        parser.add_argument('--lora_msa', type=int, nargs=3, default=[1, 0, 1])
        parser.add_argument('--lora_shared_pos', type=int, nargs='+', default=list(range(6)))
        parser.add_argument('--lora_specific_pos', type=int, nargs='+', default=list(range(6, 12)))
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if dataset is None or dataset.SETTING not in self.COMPATIBILITY:
            raise ValueError('Low-rank methods require a class-il or task-il dataset')
        if not all(hasattr(backbone, name) for name in ('update_fc', 'interface', 'image_encoder')):
            raise ValueError('Low-rank methods require --backbone low-rank-vit')
        if not isinstance(dataset.N_CLASSES_PER_TASK, int):
            raise ValueError('The source classifier pools require equal task sizes')
        if args.lr_scheduler != 'cosine' or args.scheduler_mode != 'epoch':
            raise ValueError('Low-rank methods use an epoch cosine schedule')
        if args.fitting_mode != 'epochs' or args.enable_other_metrics:
            raise ValueError('Low-rank methods require epoch fitting with task-end evaluation')
        if self.METHOD == 'cllora' and args.eval_epochs is not None:
            raise ValueError('CL-LoRA prototypes are available after end_task')
        source_transforms(dataset)
        if hasattr(dataset, 'TRANSFORM'):
            transform = transforms.Compose([transforms.ToPILImage(), dataset.TRANSFORM])
        # There is no active task before begin_task, hence no optimizer yet.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        from models.low_rank_utils.ewc import EWCLifecycle
        from models.low_rank_utils.sd import SDLifecycle
        from models.low_rank_utils.inf import InfLifecycle
        from models.low_rank_utils.cl import CLLifecycle
        lifecycle = {'ewclora': EWCLifecycle, 'sdlora': SDLifecycle,
                     'inflora': InfLifecycle, 'cllora': CLLifecycle}
        self.algorithm = lifecycle[self.METHOD]()
        self.register_buffer('_low_rank_state', torch.tensor([0, -1, 0, 0], dtype=torch.long))

    def get_parameters(self):
        return (parameter for parameter in self.net.parameters() if parameter.requires_grad)

    def get_optimizer(self, params=None, lr=None):
        if params is not None:
            return super().get_optimizer(params, lr)
        rate = self.args.lr if lr is None else lr
        if self.METHOD in ('ewclora', 'sdlora'):
            groups = [
                {'params': [p for p in self.net.image_encoder.parameters() if p.requires_grad], 'lr': rate},
                {'params': [p for p in self.net.classifier_pool.parameters() if p.requires_grad], 'lr': self.args.lora_fc_lr}]
            groups = [group for group in groups if group['params']]
        else:
            parameters = list(self.get_parameters())
            groups = [{'params': parameters, 'lr': rate}] if parameters else []
        if not groups:
            return None
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(groups, lr=rate, momentum=self.args.optim_mom,
                                   weight_decay=self.args.optim_wd, nesterov=self.args.optim_nesterov)
        optimizers = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
        return optimizers[self.args.optimizer](groups, lr=rate, weight_decay=self.args.optim_wd)

    def _restoring_prefix(self):
        return (getattr(self.args, 'loadcheck', None) is not None
                and self.current_task < (getattr(self.args, 'start_from', None) or 0))

    def begin_task(self, dataset):
        if hasattr(self.net, '_device'):
            self.net._device = self.device
        self.net.update_fc(self.n_seen_classes)
        self.net.to(self.device)
        self.algorithm.freeze(self.net, self.current_task)
        if not self._restoring_prefix():
            self.algorithm.begin_task(self, dataset)
        self.opt = self.get_optimizer()
        self._low_rank_state.copy_(torch.tensor([self.current_task, self.net._cur_task,
                                                self.n_past_classes, self.n_seen_classes],
                                               device=self._low_rank_state.device))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        mask = labels >= self.n_past_classes
        inputs, labels = inputs[mask], labels[mask] - self.n_past_classes
        if len(labels) == 0:
            return 0.
        return self.algorithm.observe(self, inputs, labels)

    def end_task(self, dataset):
        if not self._restoring_prefix():
            self.algorithm.end_task(self, dataset)
        self._low_rank_state.copy_(torch.tensor([self.current_task + 1, self.net._cur_task,
                                                self.n_seen_classes, self.n_seen_classes],
                                               device=self._low_rank_state.device))

    def _prototype_loader(self, dataset):
        source = dataset.train_loader
        view = copy.copy(source.dataset)
        if isinstance(view, MammothDatasetWrapper):
            object.__setattr__(view, 'dataset', copy.copy(source.dataset.dataset))
        view.transform = dataset.TEST_TRANSFORM
        return create_seeded_dataloader(self.args, view, batch_size=self.args.batch_size,
                                        shuffle=True, drop_last=False, pin_memory=source.pin_memory)

    def end_eval(self, dataset, accs):
        # The source evaluates EWC before accumulating A and B separately.
        # Mammoth evaluates after end_task, so defer only that merge/reset.
        if self.METHOD == 'ewclora':
            self.algorithm.finalize_task(self)

    def forward(self, inputs):
        return self.net.interface(inputs)[:, :int(self._low_rank_state[3])]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + '_low_rank_state'
        if key in state_dict:
            task, network_task, past, seen = map(int, state_dict[key])
            self._current_task = task
            self.net._cur_task = network_task
            self._n_past_classes, self._n_seen_classes = past, seen
            self._n_remaining_classes = self.N_CLASSES - seen
            self._n_classes_current_task = self.cpt
            self.algorithm.freeze(self.net, network_task)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # CL's growing prototype/proxy heads are allocated during recursive
        # loading. Apply the freeze contract after those modules exist.
        self.algorithm.freeze(self.net, self.net._cur_task)
        self.opt = self.get_optimizer()
        return result
