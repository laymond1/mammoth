import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

from utils import smart_joint, create_if_not_exists
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_validation_indexes #, get_train_val


class MyMiniDomainNet(Dataset):
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
        self.not_aug_transform = transforms.Compose(self.test_trsf + [transforms.ToTensor()])
        self.domain_names = ["clipart", "painting", "real", "sketch"]
        self.class_order = np.arange(345).tolist()

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
            
        root = os.path.dirname(os.path.dirname(self.root)) + '/domainnet/'
        imgs = []
        domains = []
        for domain_id, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
            domains += [domain_id] * len(image_list)
        data, targets = [], []
        for item in imgs:
            data.append(os.path.join(root, item[0]))
            targets.append(item[1])
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.domains = np.array(domains)
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

        # not_aug_img = self.not_aug_transform(original_img)

        # return img, target, not_aug_img
        return img, target


class SequentialMiniDomainNet(ContinualDataset):
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

    NAME = 'seq-minidomainnet'
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
    
    def set_dataset(self):
        self.train_dataset = MyMiniDomainNet(base_path() + 'mini-domainnet/splits_mini',
                                    train=True, download=False, transform=self.TRANSFORM)
        if self.args.validation:
            self.train_dataset, self.test_dataset = get_train_val(
                self.train_dataset, self.TRANSFORM, self.NAME, val_perc=self.args.validation)
        else:
            self.test_dataset = MyMiniDomainNet(base_path() + 'mini-domainnet/splits_mini',
                                train=False, download=False, transform=self.TEST_TRANSFORM)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyMiniDomainNet(base_path() + 'mini-domainnet/splits_mini',
                                   train=True, download=False, transform=self.TRANSFORM)
        test_dataset = MyMiniDomainNet(base_path() + 'mini-domainnet/splits_mini',
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
        transform = transforms.Normalize(SequentialMiniDomainNet.MEAN, SequentialMiniDomainNet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialMiniDomainNet.MEAN, SequentialMiniDomainNet.STD)
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


class ValidationDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: np.ndarray,
                 transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

def get_train_val(train: Dataset, test_transform: nn.Module,
                  dataset: str, val_perc: float = 0.1):
    """
    Extract val_perc% of the training set as the validation set.

    Args:
        train: training dataset
        test_transform: transformation of the test dataset
        dataset: dataset name
        val_perc: percentage of the training set to be extracted

    Returns:
        the training set and the validation set
    """
    dataset_length = train.data.shape[0]
    directory = 'datasets/val_permutations/'
    create_if_not_exists(directory)
    file_name = dataset + '.pt'
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length)
        torch.save(perm, directory + file_name)

    train_idxs, val_idxs = get_validation_indexes(val_perc, train)
    test_targets = np.array(train.targets)[val_idxs]

    test_dataset = ValidationDataset(train.data[val_idxs],
                                     test_targets.tolist(),
                                     transform=test_transform)
    test_dataset.classes = train.classes
    
    train.data = train.data[train_idxs]
    train_targets = np.array(train.targets)[train_idxs]
    train.targets = train_targets.tolist()

    return train, test_dataset