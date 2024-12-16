import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Tuple
from tqdm import tqdm
import json

try:
    import deeplake
except ImportError:
    raise NotImplementedError("Deeplake not installed. Please install with `pip install deeplake` to use this dataset.")

from utils import create_if_not_exists
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_validation_indexes # , get_train_val
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


def load_and_preprocess_cars196(train_str='train', names_only=False) -> Tuple[torch.Tensor, torch.Tensor, dict] | dict:
    """
    Loads data from deeplake and preprocesses it to be stored locally.

    Args:
        train_str (str): 'train' or 'test'.
        names_only (bool): If True, returns the class names only.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict] | dict: If names_only is False, returns a tuple of data, targets, and class_idx_to_name
    """
    assert train_str in ['train', 'test'], "train_str must be 'train' or 'test'"
    ds = deeplake.load(f"hub://activeloop/stanford-cars-{train_str}")
    loader = ds.pytorch()
    class_names = ds['car_models'].info['class_names']
    class_idx_to_name = {i: class_names[i] for i in range(len(class_names))}
    if names_only:
        return class_idx_to_name

    # Pre-process dataset
    data = []
    targets = []
    for x in tqdm(loader, desc=f'Pre-processing {train_str} dataset'):
        img = x['images'][0].permute(2, 0, 1)  # load one image at a time
        if len(img) < 3:
            img = img.repeat(3, 1, 1)  # fix rgb
        img = MyCars196.PREPROCESSING_TRANSFORM(img)  # resize
        data.append(img)
        label = x['car_models'][0].item()  # get label
        targets.append(label)

    data = torch.stack(data)  # stack all images
    targets = torch.tensor(targets)

    return data, targets, class_idx_to_name


class MyCars196(Dataset):
    N_CLASSES = 196

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    PREPROCESSING_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
    ])

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.ToTensor()
        self.download = download

        train_str = 'train' if train else 'test'
        if not os.path.exists(f'{root}/{train_str}_images.pt'):
            print(f'Preparing {train_str} dataset...', file=sys.stderr)
            self.load_and_preprocess_dataset(root, train_str)
        else:
            print(f"Loading pre-processed {train_str} dataset...", file=sys.stderr)
            self.data = torch.load(f'{root}/{train_str}_images.pt')
            self.targets = torch.load(f'{root}/{train_str}_labels.pt')

        self.class_names = MyCars196.get_class_names()
        # self.classes = [x for x in range(self.targets.max() + 1)]
        self.classes = list(range(self.N_CLASSES))

    def load_and_preprocess_dataset(self, root, train_str='train'):
        self.data, self.targets, class_idx_to_name = load_and_preprocess_cars196(train_str)

        print(f"Saving pre-processed dataset in {root} ({train_str}_images.pt and {train_str}_labels.py)...", file=sys.stderr)
        if not os.path.exists(root):
            os.makedirs(root)
        torch.save(self.data, f'{root}/{train_str}_images.pt')
        torch.save(self.targets, f'{root}/{train_str}_labels.pt')

        with open(f'{root}/class_names.json', 'wt') as f:
            json.dump(class_idx_to_name, f, indent=4)
        print('Done', file=sys.stderr)

    @staticmethod
    def get_class_names():
        if not os.path.exists(base_path() + f'cars196/class_names.json'):
            print("Class names not found, performing pre-processing...")
            class_idx_to_name = load_and_preprocess_cars196(names_only=True)
            print('Done', file=sys.stderr)
        else:
            with open(base_path() + f'cars196/class_names.json', 'rt') as f:
                class_idx_to_name = json.load(f)
        class_names = list(class_idx_to_name.values())
        return class_names

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCars196(ContinualDataset):
    """
    Sequential CARS196 Dataset. The images are loaded from deeplake, resized to 224x224, and store locally.
    """

    NAME = 'seq-cars196'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 196
    N_CLASSES_PER_TASK = [20] * 9 + [16]
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])  # no transform for test

    def set_dataset(self):
        self.train_dataset = MyCars196(base_path() + 'cars196', train=True,
                                    download=True, transform=self.TRANSFORM)
        if self.args.validation:
            self.train_dataset, self.test_dataset = get_train_val(
                self.train_dataset, self.TRANSFORM, self.NAME, val_perc=self.args.validation)
        else:
            self.test_dataset = MyCars196(base_path() + 'cars196', train=False,
                                download=True, transform=self.TEST_TRANSFORM)
        
    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCars196(base_path() + 'cars196', train=True,
                                  download=True, transform=self.TRANSFORM)
        test_dataset = MyCars196(base_path() + 'cars196', train=False,
                                 download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_prompt_templates():
        return templates['cars196']

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = MyCars196.get_class_names()
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCars196.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialCars196.MEAN, std=SequentialCars196.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCars196.MEAN, SequentialCars196.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

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

        img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

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