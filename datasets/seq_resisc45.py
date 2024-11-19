import os
import yaml
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_train_val
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_templates import templates


class Resisc45(Dataset):

    N_CLASSES = 45
    LABELS = [
        'airplane',
        'airport',
        'baseball_diamond',
        'basketball_court',
        'beach',
        'bridge',
        'chaparral',
        'church',
        'circular_farmland',
        'cloud',
        'commercial_area',
        'dense_residential',
        'desert',
        'forest',
        'freeway',
        'golf_course',
        'ground_track_field',
        'harbor',
        'industrial_area',
        'intersection',
        'island',
        'lake',
        'meadow',
        'medium_residential',
        'mobile_home_park',
        'mountain',
        'overpass',
        'palace',
        'parking_lot',
        'railway',
        'railway_station',
        'rectangular_farmland',
        'river',
        'roundabout',
        'runway',
        'sea_ice',
        'ship',
        'snowberg',
        'sparse_residential',
        'stadium',
        'storage_tank',
        'tennis_court',
        'terrace',
        'thermal_power_station',
        'wetland',
    ]

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()]
        )

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                # download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
                print("Downloading resisc45 dataset...")
                ln = 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EbxMu5z5HbVIkG9qFCGbg7ABDRZvpBEA8uqVC-Em9HYVug?e=Cfc4Yc'
                from onedrivedownloader import download
                download(ln, filename=os.path.join(root, 'resisc45.tar.gz'), unzip=True, unzip_path=root, clean=True)
                print("Done!")

        if self.train:
            data_config = yaml.load(open(smart_joint(root, 'resisc45_train.yaml')), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(smart_joint(root, 'resisc45_test.yaml')), Loader=yaml.Loader)

        self.data = np.array([smart_joint(root, d) for d in data_config['data']])
        self.targets = np.array(data_config['targets']).astype(np.int64)
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

        img = Image.open(img).convert('RGB')

        original_img = img.copy()

        # not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            # return img, target, not_aug_img, self.logits[index]
            return img, target, self.logits[index]

        # return img, target, not_aug_img
        return img, target


class SequentialResisc45(ContinualDataset):

    NAME = 'seq-resisc45'
    SETTING = 'class-il'
    N_TASKS = 9
    N_CLASSES_PER_TASK = 45 // N_TASKS
    N_CLASSES = 45
    SIZE = (224, 224)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def set_dataset(self):
        self.train_dataset = Resisc45(base_path() + 'NWPU-RESISC45', train=True,
                                        download=True, transform=self.TRANSFORM)
        if self.args.validation:
            self.train_dataset, self.test_dataset = get_train_val(
                self.train_dataset, self.TRANSFORM, self.NAME, val_perc=self.args.validation)
        else:
            self.test_dataset = Resisc45(base_path() + 'NWPU-RESISC45', train=False,
                                    download=True, transform=self.TEST_TRANSFORM)

    def get_data_loaders(self):
        train_dataset = Resisc45(base_path() + 'NWPU-RESISC45', train=True,
                                 download=True, transform=self.TRANSFORM)
        test_dataset = Resisc45(base_path() + 'NWPU-RESISC45', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = [x.replace('_', ' ') for x in Resisc45.LABELS]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return classes

    @staticmethod
    def get_prompt_templates():
        return templates['eurosat']

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(),
                                   SequentialResisc45.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialResisc45.MEAN, std=SequentialResisc45.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(mean=SequentialResisc45.MEAN, std=SequentialResisc45.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
