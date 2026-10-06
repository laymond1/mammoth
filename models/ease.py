"""EASE: expandable task adapter subspaces with semantic prototype completion.

Reference: CVPR24-Ease revision 6d67508ed8b7c4d23350ed4eb28c123914b40ae3.
The standard replay-free EASE path is ported; optional reference ablations that
access old training images or add an unadapted PTM space are not enabled here.
"""

import copy
import math

import torch
from torch.utils.data import DataLoader

from backbone.ease_vit import EaseViT
from datasets.utils.continual_dataset import MammothDatasetWrapper
from models.ease_utils.network import EaseNetwork
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type


class Ease(ContinualModel):
    NAME = 'ease'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(backbone='ease-vit', optimizer='sgd', lr=0.025,
                            optim_mom=0.9, optim_wd=0.0005, batch_size=48,
                            n_epochs=20, lr_scheduler='cosine', enable_other_metrics=False)
        # Also declared by the backbone factory: dynamic argument injection skips
        # names already present on the model parser.
        parser.add_argument('--vit_size', choices=['tiny', 'small', 'base'], default='base')
        parser.add_argument('--ease_pretrain_type', choices=['in21k', 'in21k_ft_in1k'],
                            default='in21k_ft_in1k')
        parser.add_argument('--adapter_size', type=int, default=64)
        parser.add_argument('--ease_alpha', type=float, default=0.1)
        parser.add_argument('--ease_recalc_sim', type=binary_to_boolean_type, default=True)
        parser.add_argument('--ease_use_reweight', type=binary_to_boolean_type, default=True)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if not isinstance(backbone, EaseViT):
            raise ValueError('EASE requires --backbone ease-vit')
        if not backbone.pretrained_loaded:
            raise ValueError('EASE requires pretrained weights or an explicit --ease_checkpoint')
        if not math.isfinite(args.ease_alpha) or args.ease_alpha < 0:
            raise ValueError('ease_alpha must be finite and nonnegative')
        if args.fitting_mode != 'epochs' or args.eval_epochs is not None:
            raise ValueError('EASE evaluates prototypes after end_task; use epoch fitting without per-epoch evaluation')
        if args.enable_other_metrics:
            raise ValueError('EASE pre-task/epoch prototype evaluation is unsupported; disable enable_other_metrics')
        if args.label_perc != 1 or args.label_perc_by_class != 1:
            raise ValueError('EASE prototype fitting requires labels for every training sample')
        network = EaseNetwork(backbone, args.ease_alpha, args.ease_recalc_sim, args.ease_use_reweight)
        super().__init__(network, loss, args, transform, dataset=dataset)

    def get_parameters(self):
        return (parameter for parameter in self.net.parameters() if parameter.requires_grad)

    def begin_task(self, dataset):
        self.net.begin_task(self.n_past_classes, self.n_seen_classes)
        self.net.to(self.device)
        self.net.train()
        # The native scheduler helper reloads this optimizer each task as usual.
        self.opt = self.get_optimizer()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        logits = self.net.training_logits(inputs)
        loss = self.loss(logits, labels - self.n_past_classes)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def _prototype_loader(self, dataset):
        """Use current training images with test transforms, without mutation.

        A separate loader visits every sample, even when training uses drop_last
        or a replacement sampler. Only small dataset objects are shallow-copied;
        their image storage is shared. The native wrapper and label mapping stay.
        """
        source = dataset.train_loader
        view = copy.copy(source.dataset)
        if isinstance(view, MammothDatasetWrapper):
            object.__setattr__(view, 'dataset', copy.copy(source.dataset.dataset))
        view.transform = dataset.TEST_TRANSFORM
        return DataLoader(view, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, drop_last=False,
                          pin_memory=source.pin_memory)

    @torch.no_grad()
    def end_task(self, dataset):
        self.net.eval()
        new_classes = self.n_seen_classes - self.n_past_classes
        sums = torch.zeros(new_classes, self.net.classifier.in_features, device=self.device)
        counts = torch.zeros(new_classes, dtype=torch.long, device=self.device)
        for batch in self._prototype_loader(dataset):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            labels = labels - self.n_past_classes
            if ((labels < 0) | (labels >= new_classes)).any():
                raise ValueError('EASE prototypes may only use the current task training classes')
            features = self.net.backbone.extract_all(inputs)
            sums.index_add_(0, labels, features.to(sums.dtype))
            counts.index_add_(0, labels, torch.ones_like(labels))
        if (counts == 0).any():
            raise ValueError('Every new class needs training samples for its EASE prototype')
        # Average raw features, as in the reference; normalize only for cosine.
        self.net.finalize_task(sums / counts[:, None], self.n_past_classes)

    def forward(self, inputs):
        return self.net(inputs)
