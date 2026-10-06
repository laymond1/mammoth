import open_clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from PIL import Image
from typing import Tuple

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order, store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates


class MySVHN(SVHN):
    """
    Custom SVHN dataset that returns both augmented and non-augmented images.
    """

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.root = root
        super(MySVHN, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert("RGB")
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialSVHN(ContinualDataset):
    """
    The Sequential SVHN dataset with 224x224 resolution with ViT-B/16.
    """

    NAME = 'seq-svhn'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2  # Example: split into 2 tasks
    N_TASKS = 5
    N_CLASSES = 10
    SIZE = (224, 224)
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args)
        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        MEAN, STD = self.MEAN, self.STD

        transform = self.TRANSFORM

        # CLIP's official test transform
        _, _, test_transform = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir='checkpoints/ViT-B-16/cachedir/open_clip')

        train_dataset = MySVHN(root=base_path() + 'SVHN', split='train', download=True, transform=transform)
        test_dataset = MySVHN(root=base_path() + 'SVHN', split='test', download=True, transform=test_transform)

        train_dataset.targets = train_dataset.labels
        test_dataset.targets = test_dataset.labels

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialSVHN.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialSVHN.MEAN, SequentialSVHN.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialSVHN.MEAN, SequentialSVHN.STD)

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = [str(i) for i in range(10)]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates.get('svhn', templates['default'])
