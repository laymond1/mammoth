import os
import open_clip
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates
from PIL import Image

class MyGTSRB(torchvision.datasets.GTSRB):
    """
    Custom GTSRB dataset that returns both augmented and non-augmented images.
    """

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.root = root
        super(MyGTSRB, self).__init__(root, split=split, transform=transform,
                                      target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Returns a tuple containing:
        - the transformed image,
        - the target class,
        - the non-augmented (tensor) version of the original image.
        If `self.logits` exists, also returns the corresponding logits.
        """
        path, target = self.data[index], self.targets[index]
        img = Image.open(path).convert("RGB")
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def __len__(self) -> int:
        if hasattr(self, 'data'):
            return len(self.data)
        return super().__len__()
    

class SequentialGTSRB(ContinualDataset):
    """
    The Sequential GTSRB dataset with 224x224 resolution with ViT-B/16.
    """

    NAME = 'seq-gtsrb'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [10] * 4 + [3]
    N_TASKS = 5
    N_CLASSES = 43
    SIZE = (224, 224)
    MEAN = (0.48145466, 0.4578275, 0.40821073)  # from CLIP
    STD = (0.26862954, 0.26130258, 0.27577711)  # from CLIP

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
        MEAN = self.MEAN
        STD = self.STD

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        # CLIP's official test transform
        _, _, test_transform = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir='checkpoints/ViT-B-16/cachedir/open_clip')

        # Load GTSRB dataset
        train_dataset = MyGTSRB(root=os.path.join(base_path(), 'GTSRB'), split='train', download=True, transform=transform)
        test_dataset = MyGTSRB(root=os.path.join(base_path(), 'GTSRB'), split='test', download=True, transform=test_transform)

        train_dataset.data = np.array([t[0] for t in train_dataset._samples])
        train_dataset.targets = np.array([t[1] for t in train_dataset._samples])

        test_dataset.data = np.array([t[0] for t in test_dataset._samples])
        test_dataset.targets = np.array([t[1] for t in test_dataset._samples])

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialGTSRB.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialGTSRB.MEAN, SequentialGTSRB.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialGTSRB.MEAN, SequentialGTSRB.STD)

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        classes = [  # Classes used in Ilharco et al 2023
            'red and white circle 20 kph speed limit',
            'red and white circle 30 kph speed limit',
            'red and white circle 50 kph speed limit',
            'red and white circle 60 kph speed limit',
            'red and white circle 70 kph speed limit',
            'red and white circle 80 kph speed limit',
            'end / de-restriction of 80 kph speed limit',
            'red and white circle 100 kph speed limit',
            'red and white circle 120 kph speed limit',
            'red and white circle red car and black car no passing',
            'red and white circle red truck and black car no passing',
            'red and white triangle road intersection warning',
            'white and yellow diamond priority road',
            'red and white upside down triangle yield right-of-way',
            'stop',
            'empty red and white circle',
            'red and white circle no truck entry',
            'red circle with white horizonal stripe no entry',
            'red and white triangle with exclamation mark warning',
            'red and white triangle with black left curve approaching warning',
            'red and white triangle with black right curve approaching warning',
            'red and white triangle with black double curve approaching warning',
            'red and white triangle rough / bumpy road warning',
            'red and white triangle car skidding / slipping warning',
            'red and white triangle with merging / narrow lanes warning',
            'red and white triangle with person digging / construction / road work warning',
            'red and white triangle with traffic light approaching warning',
            'red and white triangle with person walking warning',
            'red and white triangle with child and person walking warning',
            'red and white triangle with bicyle warning',
            'red and white triangle with snowflake / ice warning',
            'red and white triangle with deer warning',
            'white circle with gray strike bar no speed limit',
            'blue circle with white right turn arrow mandatory',
            'blue circle with white left turn arrow mandatory',
            'blue circle with white forward arrow mandatory',
            'blue circle with white forward or right turn arrow mandatory',
            'blue circle with white forward or left turn arrow mandatory',
            'blue circle with white keep right arrow mandatory',
            'blue circle with white keep left arrow mandatory',
            'blue circle with white arrows indicating a traffic circle',
            'white circle with gray strike bar indicating no passing for cars has ended',
            'white circle with gray strike bar indicating no passing for trucks has ended',
        ]
 
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names


    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']
