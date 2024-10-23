import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils import smart_joint
from utils.conf import base_path
from datasets.utils import set_default_from_args


class MyDomainNet(Dataset):
    """Defines the iDomainNet dataset."""

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                raise NotImplementedError("Download not implemented yet.")
                # Placeholder for actual download logic

        self._load_data()

    def _load_data(self):
        if self.train:
            image_list_paths = [os.path.join(self.root, d + "_train.txt") for d in self.domain_names]
        else:
            image_list_paths = [os.path.join(self.root, d + "_test.txt") for d in self.domain_names]
            
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        data, targets = [], []
        for item in imgs:
            data.append(os.path.join(self.root, item[0]))
            targets.append(item[1])
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.classes = [x for x in range(self.targets.max() + 1)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        not_aug_img = self.not_aug_transform(original_img)

        return img, target, not_aug_img


class SequentialDomainNet(ContinualDataset):
    """The Sequential iDomainNet dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-domainnet'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 345
    N_TASKS = 5
    N_CLASSES = 345
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    SIZE = (224, 224)
    TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    
    train_dataset = MyDomainNet(base_path() + 'domainnet',
                                   train=True, download=False, transform=TRANSFORM)
    test_dataset = MyDomainNet(base_path() + 'domainnet',
                                train=False, download=False, transform=TEST_TRANSFORM)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyDomainNet(base_path() + 'domainnet',
                                   train=True, download=False, transform=self.TRANSFORM)
        test_dataset = MyDomainNet(base_path() + 'domainnet',
                                  train=False, download=False, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialDomainNet.MEAN, SequentialDomainNet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialDomainNet.MEAN, SequentialDomainNet.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        CLASS_NAMES = os.listdir(base_path() + 'domainnet/clipart')
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names
