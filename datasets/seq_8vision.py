from typing import Tuple
from itertools import chain
import logging
from copy import deepcopy
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.seq_cars196 import SequentialCars196
from datasets.seq_dtd import SequentialDTD
from datasets.seq_eurosat_rgb import SequentialEuroSatRgb
from datasets.seq_gtrsrb import SequentialGTSRB
from datasets.seq_mnist_224 import SequentialMNIST224
from datasets.seq_resisc45 import SequentialResisc45
from datasets.seq_sun397 import SequentialSUN397
from datasets.seq_svhn import SequentialSVHN
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates


class Sequential8Vision(ContinualDataset):
    """
    Sequential 8 Vision dataset. Each task is a different vision dataset, and the model is trained on them sequentially.
    
    The datasets are: Cars196, DTD, EuroSat RGB, GTSRB, MNIST-224, Resisc45, SUN397 and SVHN.
    """
    NAME = 'seq-8vision'
    SETTING = 'class-il'
    DATASET_NAMES = ["seq-cars196", "seq-dtd", "seq-eurosat-rgb", "seq-gtsrb", "seq-mnist-224", "seq-resisc45", "seq-sun397", "seq-svhn"]
    DATASETS = [SequentialCars196, SequentialDTD, SequentialEuroSatRgb, SequentialGTSRB, SequentialMNIST224, SequentialResisc45, SequentialSUN397, SequentialSVHN]
    N_CLASSES_PER_TASK = [196, 47, 10, 43, 10, 45, 397, 10]
    N_TASKS = 8
    N_CLASSES = sum(N_CLASSES_PER_TASK)
    SIZE = (224, 224)
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    TEST_TRANSFORM = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='openai', cache_dir='checkpoints/ViT-B-16/cachedir/open_clip')[2]


    def __init__(self, args):
        super().__init__(args)
        self.dataset_instances = []
        args_joint = deepcopy(args)
        args_joint.joint = True
        for dataset in self.DATASETS:
            self.dataset_instances.append(dataset(args_joint))
            self.dataset_instances[-1].TRANSFORM = self.TRANSFORM
            self.dataset_instances[-1].TEST_TRANSFORM = self.TEST_TRANSFORM
        self.test_loaders = []

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        self.c_task += 1

        cur_dataset = self.dataset_instances[self.c_task]
        self.train_loader, test_loader = cur_dataset.get_data_loaders()
        self.train_loader.dataset.targets += sum(self.N_CLASSES_PER_TASK[:self.c_task])
        test_loader.dataset.targets += sum(self.N_CLASSES_PER_TASK[:self.c_task])
        self.test_loaders.append(test_loader)
        return self.train_loader, self.test_loaders

    @staticmethod
    def get_transform():
        return None

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        if self.args.n_epochs is not None:
            return self.args.n_epochs
        return self.dataset_instances[self.c_task].get_epochs()

    def get_task_epochs(self, t):
        epochs = {
            "seq-cars196": 35,
            "seq-dtd": 76,
            "seq-eurosat-rgb": 12,
            "seq-gtsrb": 11,
            "seq-mnist-224": 5,
            "seq-resisc45": 15,
            "seq-sun397": 14,
            "seq-svhn": 4,
        }
        return epochs[self.DATASET_NAMES[t]]
    
    def get_iters(self):
        iters = 2000
        if self.args.chunks is not None:
            iters *= self.args.chunks
        return iters

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = list(chain.from_iterable(
            dataset.get_class_names() for dataset in self.dataset_instances
        ))
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['seq-8vision']
