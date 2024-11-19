import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import set_default_from_args
from utils import smart_joint, create_if_not_exists
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_validation_indexes #, get_train_val
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class Isic(Dataset):
    N_CLASSES = 6

    LABELS = ['melanoma',
              'basal cell carcinoma',
              'actinic keratosis or intraepithelial carcinoma',
              'benign keratosis',
              'dermatofibroma',
              'vascular skin lesion']

    """
    Overrides the ChestX dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        split = 'train' if train else 'test'
        if not os.path.exists(f'{root}/{split}_images.pkl'):
            if download:
                ln = 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/ERM64PkPkFtJhmiUQkVvE64BR900MbIHtJVA_CR4KKhy8A?e=OsrQr5'
                from onedrivedownloader import download
                download(ln, filename=smart_joint(root, 'isic.tar.gz'), unzip=True, unzip_path=root.rstrip('isic'), clean=True)
            else:
                raise FileNotFoundError(f'File not found: {root}/{split}_images.pkl')

        filename_labels = f'{self.root}/{split}_labels.pkl'
        filename_images = f'{self.root}/{split}_images.pkl'

        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

        with open(filename_images, 'rb') as f:
            self.data = pickle.load(f)

        with open(filename_labels, 'rb') as f:
            self.targets = pickle.load(f)
        
        self.classes = [x for x in range(self.targets.max()+1)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray((img * 255).astype(np.int8), mode='RGB')

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialIsic(ContinualDataset):

    NAME = 'seq-isic'
    SETTING = 'class-il'
    N_TASKS = 3
    N_CLASSES_PER_TASK = 2
    N_CLASSES = 6
    SIZE = (224, 224)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE[0]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def set_dataset(self):
        self.train_dataset = Isic(base_path() + 'isic', train=True,
                                  download=True, transform=self.TRANSFORM)
        if self.args.validation:
            self.train_dataset, self.test_dataset = get_train_val(
                self.train_dataset, self.TRANSFORM, self.NAME, val_perc=self.args.validation)
        else:
            self.test_dataset = Isic(base_path() + 'isic', train=False,
                                     download=True, transform=self.TEST_TRANSFORM)

    def get_data_loaders(self):
        train_dataset = Isic(base_path() + 'isic', train=True,
                             download=True, transform=self.TRANSFORM)

        test_dataset = Isic(base_path() + 'isic', train=False, download=True,
                            transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(Isic.LABELS, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            SequentialIsic.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialIsic.MEAN, std=SequentialIsic.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean=SequentialIsic.MEAN, std=SequentialIsic.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128


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

        img = Image.fromarray((img * 255).astype(np.int8), mode='RGB')

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