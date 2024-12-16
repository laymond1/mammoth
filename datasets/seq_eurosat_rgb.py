import io
import json
import logging
import os
import sys
import zipfile
import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except ImportError:
    raise ImportError("Please install the google_drive_downloader package by running: `pip install googledrivedownloader`")

from datasets.utils import set_default_from_args
from utils import create_if_not_exists
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_validation_indexes #, get_train_val
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class MyEuroSat(Dataset):

    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.split = split
        assert split in ['train', 'test', 'val'], 'Split must be either train, test or val'
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.ToTensor()

        if not os.path.exists(root + '/DONE'):
            print('Preparing dataset...', file=sys.stderr)
            r = requests.get('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(root)
            os.system(f'mv {root}/EuroSAT_RGB/* {root}')
            os.system(f'rmdir {root}/EuroSAT_RGB')

            # create DONE file
            with open(self.root + '/DONE', 'w') as f:
                f.write('')

            # downlaod split file form https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/
            # from "Conditional Prompt Learning for Vision-Language Models", Kaiyang Zhou et al.
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=self.root + '/split.json')

            print('Done', file=sys.stderr)

        self.data_split = pd.DataFrame(json.load(open(self.root + '/split.json', 'r'))[split])
        self.data = self.data_split[0].values
        self.targets = self.data_split[1].values
        self.classes = [x for x in range(self.targets.max()+1)]

        self.class_names = self.get_class_names()

    @staticmethod
    def get_class_names():
        if not os.path.exists(base_path() + f'eurosat/DONE'):
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=base_path() + 'eurosat/split.json')
        return pd.DataFrame(json.load(open(base_path() + 'eurosat/split.json', 'r'))['train'])[2].unique()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(self.root + '/' + img).convert('RGB')

        not_aug_img = self.totensor(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split != 'train':
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


def my_collate_fn(batch):
    tmp = list(zip(*batch))
    imgs = torch.stack(tmp[0], dim=0)
    labels = torch.tensor(tmp[1])
    if len(tmp) == 2:
        return imgs, labels
    not_aug_imgs = tmp[2]
    not_aug_imgs = torch.stack(not_aug_imgs, dim=0)
    if len(tmp) == 4:
        logits = torch.stack(tmp[3], dim=0)
        return imgs, labels, not_aug_imgs, logits
    return imgs, labels, not_aug_imgs


class SequentialEuroSatRgb(ContinualDataset):

    NAME = 'seq-eurosat-rgb'
    SETTING = 'class-il'
    N_TASKS = 5
    N_CLASSES = 10
    N_CLASSES_PER_TASK = 2
    SIZE = (224, 224)
    MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),  # from https://github.dev/KaiyangZhou/Dassl.pytorch defaults
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(SIZE[0], interpolation=InterpolationMode.BICUBIC),  # bicubic
        transforms.CenterCrop(SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def set_dataset(self):
        self.train_dataset = MyEuroSat(base_path() + 'eurosat', split='train',
                                       transform=self.TRANSFORM)
        if self.args.validation:
            self.train_dataset, self.test_dataset = get_train_val(
                self.train_dataset, self.TRANSFORM, self.NAME, val_perc=self.args.validation)
        else:
            self.test_dataset = MyEuroSat(base_path() + 'eurosat', split='test',
                                          transform=self.TEST_TRANSFORM)

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        try:
            classes = MyEuroSat.get_class_names()
        except BaseException:
            logging.info("dataset not loaded yet -- loading dataset...")
            MyEuroSat(base_path() + 'eurosat', train=True,
                                    transform=None)
            classes = MyEuroSat.get_class_names()

        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    def get_data_loaders(self):
        train_dataset = MyEuroSat(base_path() + 'eurosat', split='train',
                                  transform=self.TRANSFORM)
        test_dataset = MyEuroSat(base_path() + 'eurosat', split='test',
                                 transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose([transforms.ToPILImage(),
                                        SequentialEuroSatRgb.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialEuroSatRgb.MEAN, std=SequentialEuroSatRgb.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialEuroSatRgb.MEAN, SequentialEuroSatRgb.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    @staticmethod
    def get_prompt_templates():
        return templates['eurosat']


class ValidationDataset(Dataset):
    def __init__(self, root, data: torch.Tensor, targets: np.ndarray,
                 transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None) -> None:
        self.root = root
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.open(self.root + '/' + img).convert('RGB')

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

    test_dataset = ValidationDataset(train.root,
                                     train.data[val_idxs],
                                     test_targets.tolist(),
                                     transform=test_transform)
    test_dataset.classes = train.classes
    
    train.data = train.data[train_idxs]
    train_targets = np.array(train.targets)[train_idxs]
    train.targets = train_targets.tolist()

    return train, test_dataset