from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST

from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.transforms.denormalization import DeNormalize


class MyMNIST224(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        super().__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L').convert("RGB")
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class SequentialMNIST224(ContinualDataset):
    """
    Sequential MNIST dataset in 224x224 RGB for CLIP.
    """

    NAME = 'seq-mnist-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (224, 224)
    MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP mean
    STD = (0.26862954, 0.26130258, 0.27577711)  # CLIP std

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyMNIST224(base_path() + 'MNIST',
                                train=True, download=True, transform=self.TRANSFORM)
        test_dataset = MyMNIST224(base_path() + 'MNIST',
                               train=False, download=True, transform=self.TEST_TRANSFORM)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_transform():
        return transforms.Compose(
            [transforms.ToPILImage(), SequentialMNIST224.TRANSFORM])

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialMNIST224.MEAN, SequentialMNIST224.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialMNIST224.MEAN, SequentialMNIST224.STD)

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 20

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = [str(i) for i in range(10)]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
