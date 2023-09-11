# This code is taken from : https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cub2011.py
# This code is adjusted by Wonseon Lim.

from typing import Tuple, Optional

import os

import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class CUB200(Dataset):
    """
    Defines CUB200 as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                import tarfile
                print('Downloading dataset')
                download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

                with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
                    tar.extractall(path=self.root)
                    
                self._load_metadata()
                if self.data is None:
                    self._save_data()

        self.data = np.load(os.path.join(
            root, 'processed/x_CUB_200_2011_%s.npy' %
                    ('train' if self.train else 'test')))

        self.targets = np.load(os.path.join(
            root, 'processed/y_CUB_200_2011_%s.npy' %
                    ('train' if self.train else 'test')))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data_df = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data_df = self.data_df[self.data_df.is_training_img == 1]
        else:
            self.data_df = self.data_df[self.data_df.is_training_img == 0]
        # Targets start at 1 by default, so shift to 0
        self.targets = self.data_df[:, 2] - 1 
        
    def _save_data(self):
        self.data = []
        # img_id / filepath / target / is_training_img
        for index, row in self.data_df.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            img = np.array(default_loader(filepath))
            img = cv2.resize(img, dsize=(128, 128))
            self.data.append(img)
        self.data = np.array(self.data)
        if not os.path.exists(os.path.join(self.root, 'processed')):
            os.makedirs(os.path.join(self.root, 'processed'))
        np.save(os.path.join(self.root, 'processed/x_CUB_200_2011_%s.npy' %
                ('train' if self.train else 'test')), self.data)
        np.save(os.path.join(self.root, 'processed/y_CUB_200_2011_%s.npy' %
                ('train' if self.train else 'test')), self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[idx]

        return img, target
        

class MyCUB200(CUB200):
    """
    Overrides the CUB200 dataset to change the getitem function.
    """
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(MyCUB200, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    
class SequentialCUB200(ContinualDataset):

    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.Resize((128, 128)),
             transforms.RandomCrop(128, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                  (0.2023, 0.1994, 0.2010))])

    def get_examples_number(self):
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True, 
                                 download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((128, 128)),
             transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCUB200(base_path() + 'CUB200', train=True, 
                                 download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CUB200(base_path() + 'CUB200', train=False, 
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCUB200.N_CLASSES_PER_TASK
                        * SequentialCUB200.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                         (0.2023, 0.1994, 0.2010))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCUB200.get_batch_size()

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_image_size():
        return 128