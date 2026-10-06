
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

class MyDTD(torchvision.datasets.DTD):
    """
    Custom DTD dataset that returns both augmented and non-augmented images.
    """

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.root = root
        super(MyDTD, self).__init__(root, split=split, transform=transform,
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
        #img = self.loader(path)
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
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
    

class SequentialDTD(ContinualDataset):
    """
    The Sequential CIFAR100 dataset with 224x224 resolution with ViT-B/16.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """

    NAME = 'seq-dtd'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [10] * 4 + [7]
    N_TASKS = 5
    N_CLASSES = 47
    SIZE = (224, 224)
    MEAN = (0.48145466, 0.4578275, 0.40821073) # from clip
    STD = (0.26862954, 0.26130258, 0.27577711) # from clip

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
        MEAN = (0.48145466, 0.4578275, 0.40821073)
        STD = (0.26862954, 0.26130258, 0.27577711)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        _, _, test_transform = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir='checkpoints/ViT-B-16/cachedir/open_clip')

        # Load DTD dataset
        train_dataset = MyDTD(root=base_path() + 'DTD', split='train', download=True, transform=transform)
        test_dataset = MyDTD(root=base_path() + 'DTD', split='val', download=True, transform=test_transform)

        for dataset in [train_dataset, test_dataset]:
            dataset.data = np.array(dataset._image_files)
            dataset.targets = np.array(dataset._labels)

        # Use your masking-aware loader method
        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test


    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialDTD.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialDTD.MEAN, SequentialDTD.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialDTD.MEAN, SequentialDTD.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        from torchvision.datasets import DTD
        classes = DTD(base_path() + 'DTD', split='train', download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']