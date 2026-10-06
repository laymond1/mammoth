import importlib
import json
import os
import site
import sys
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple

from tqdm import tqdm

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order, store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates


class custom_import:
    """
    Context manager for importing packages with the same name while avoiding conflicts.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name

    def load_package(self, package_name: str, basepath: str):
        package_path = os.path.join(basepath, f'{package_name}/__init__.py')
        spec = importlib.util.spec_from_file_location(package_name, package_path)
        pkg_datasets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pkg_datasets)

    def clean_cache(self):
        # reset the import system
        for key in list(sys.modules.keys()):
            if key.startswith(self.module_name):
                del sys.modules[key]
        importlib.invalidate_caches()

    def rename_cache(self, old_module_name: str, new_module_name: str):
        # reset the import system
        for key in list(sys.modules.keys()):
            if key.startswith(old_module_name):
                sys.modules[key.replace(old_module_name, new_module_name)] = sys.modules.pop(key)
                # del sys.modules[key]

    def __enter__(self):
        """
        Stores the current state of the import system and loads the module ignoring cache.
        """

        self.original_modules = sys.modules.copy()
        self.original_syspath = sys.path.copy()

        # reset the sys path with only the internal paths
        sys.path = [p for p in sys.path if 'site-packages' in p]

        # move the current module to a temporary name
        self.rename_cache(self.module_name, 'pre_cached_' + self.module_name)

        if any(['old_cached_' + self.module_name in module for module in sys.modules.keys()]):
            # if the module is already loaded, rename it and serve it
            self.rename_cache('old_cached_' + self.module_name, self.module_name)
        else:
            # if the module is not loaded, load it
            self.load_package(self.module_name, site.getsitepackages()[0])

        self.imported_module = importlib.import_module(self.module_name)
        return self.imported_module

    def __exit__(self, type, value, traceback):
        """
        Restores the import system to the previous state.
        """
        # save the imported module
        self.rename_cache(self.module_name, 'old_cached_' + self.module_name)

        # swap the original module back
        self.rename_cache('pre_cached_' + self.module_name, self.module_name)

        sys.path = self.original_syspath

        importlib.import_module(self.module_name)


def download_SUN(force_download=False):
    if os.path.isfile(os.path.join(base_path(), "SUN397/done.flag")) and not force_download:
        print("dataset already on disk")
        with open(os.path.join(base_path(), "SUN397", "class_names.json")) as f:
            class_names = json.load(f)["class_names"]
        return class_names
    
    with custom_import("datasets") as huggingface_datasets:
        hf_full = huggingface_datasets.load_dataset("tanganke/sun397", split="train", cache_dir=base_path() + "huggingface_datasets")
        class_names = hf_full._info.features["label"].names
        os.makedirs(os.path.join(base_path(), "SUN397"), exist_ok=True)
        with open(os.path.join(base_path(), "SUN397", "class_names.json"), "w") as f:
            json.dump({"class_names": class_names}, f, indent=4)

        # Create a deterministic train/test split from the HF 'train' split (common HF variants provide a single split)
        split = hf_full.train_test_split(test_size=0.20, seed=42)
        hf_train = split["train"]
        hf_test = split["test"]

    def save_split(hf_dataset, split_name):
        split_root_dir = os.path.join(base_path(), "SUN397", split_name)
        os.makedirs(split_root_dir, exist_ok=True)
        mapping = {}

        print(f"Saving SUN {split_name} on disk")
        for i, example in enumerate(tqdm(hf_dataset)):
            image = example['image'].convert('RGB')
            label = example['label']
            class_dir = os.path.join(split_root_dir, class_names[label])
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{split_name}_{i}.jpg"
            filepath = os.path.join(class_dir, filename)
            image.save(filepath)

            mapping[os.path.join(class_dir, filename)] = label

        with open(os.path.join(split_root_dir, f"{split_name}_labels.json"), "w") as f:
            json.dump(mapping, f, indent=4)

    save_split(hf_train, "train")
    save_split(hf_test, "val")
    open(os.path.join(base_path(), "SUN397", "done.flag"), "w").close()
    return class_names


class MySUN397(torch.utils.data.Dataset):
    
    def __init__(self, root, split, transform=None, target_transform=None):
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.root = root
        download_SUN()
        super(MySUN397, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(self.root, f"{split}_labels.json")) as f:
            data = json.load(f)
            self.samples = list(data.keys())
            self.labels = list(data.values())

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        path = self.data[index]
        target = self.targets[index]
        img = Image.open(path).convert("RGB")
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialSUN397(ContinualDataset):
    NAME = 'seq-sun397'
    SETTING = 'class-il'
    # SUN397 has 397 classes; adapt as needed for tasks
    N_CLASSES_PER_TASK = [50] * 7 + [47]  # Example splitting into 8 tasks (adjust as you want)
    N_TASKS = len(N_CLASSES_PER_TASK)
    N_CLASSES = 397
    SIZE = (224, 224)

    # Use CLIP mean/std
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

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
        MEAN = self.MEAN
        STD = self.STD

        transform = self.TRANSFORM

        # CLIP test transform
        _, _, test_transform = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir='checkpoints/ViT-B-16/cachedir/open_clip')

        sun_root = base_path() + 'SUN397'

        train_dataset = MySUN397(root=sun_root + '/train', split="train", transform=transform)
        test_dataset = MySUN397(root=sun_root + '/val', split="val", transform=test_transform)

        train_dataset.data = np.array(train_dataset.samples)
        train_dataset.targets = np.array(train_dataset.labels)

        test_dataset.data = np.array(test_dataset.samples)
        test_dataset.targets = np.array(test_dataset.labels)

        # Store masked loaders for continual learning setup
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

        return train_loader, test_loader

    @staticmethod
    def get_transform():
        return SequentialSUN397.TRANSFORM

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialSUN397.MEAN, SequentialSUN397.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialSUN397.MEAN, SequentialSUN397.STD)

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 128

    def get_class_names(self):
        if self.class_names is None:
            self.class_names = download_SUN()
        classes = fix_class_names_order(self.class_names, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        # You might want to create or adapt templates for SUN397 classes
        return templates.get('sun397', templates['default'])