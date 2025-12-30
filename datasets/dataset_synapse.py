import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        base_dir: 存放 .npz 的資料夾
        list_dir: 存放 train.txt / val.txt / test.txt
        split: 'train' | 'val' | 'test'
        transform: optional torch-style transform
        """
        self.transform = transform
        self.split = split
        self.data_dir = base_dir

        list_path = os.path.join(list_dir, split + ".txt")
        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name + ".npz")

        data = np.load(data_path)
        image = data["image"]      # (H, W)
        label = data["label"]      # (H, W)

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            # image: (H, W) -> (1, H, W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            # label: (H, W), class index
            label = torch.from_numpy(label.astype(np.int64))

            sample = {"image": image, "label": label}

        # 保留 slice 名稱，對 debug / 存檔很有用
        sample["case_name"] = slice_name
        return sample
