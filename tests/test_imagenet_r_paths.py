"""ImageNet-R images are found under the dataset root, not under the working directory."""

import os

from datasets.seq_imagenet_r import MyImagenetR


def test_images_are_read_from_the_root(tmp_path):
    root = tmp_path / 'imagenet-r'
    root.mkdir()
    for train in (True, False):
        dataset = MyImagenetR(str(root) + '/', train=train)
        assert len(dataset.data) == len(dataset.targets) > 0
        assert all(path.startswith(str(root) + '/n') and path.count('imagenet-r/') == 1 for path in dataset.data)
        synset, name = dataset.data[0][len(str(root)) + 1:].split('/')
        assert synset.startswith('n') and os.path.splitext(name)[1] == '.jpg'
