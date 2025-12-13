import os
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import VisionDataset

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def load_medical_image(path):
    if path.endswith('.mhd') or path.endswith('.nii') or path.endswith('.nii.gz'):
        if not HAS_SITK:
            raise RuntimeError("SimpleITK required")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}")
        return Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        return Image.open(path).convert('L')


class ImageListDataset(VisionDataset):
    def __init__(self, data_root, listfile, transform, nolabel=False, multiclass=False):
        self.image_list = []
        self.label_list = []
        self.nolabel = nolabel
        self.data_root = data_root

        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                items = line.split()
                if nolabel:
                    image_path = os.path.join(data_root, line)
                else:
                    image_path = os.path.join(data_root, items[0])
                    if not multiclass:
                        label = int(items[1])
                    else:
                        label = list(map(float, items[1:]))
                    self.label_list.append(label)
                self.image_list.append(image_path)

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]

        if img_path.endswith('.mhd') or img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
            image = load_medical_image(img_path)
        else:
            image = Image.open(img_path).convert('L')

        image = self.transform(image)

        if image.shape[0] != 1:
            raise ValueError(f"Expected 1 channel, got {image.shape}")

        if not self.nolabel:
            label = self.label_list[index]
            return image, torch.tensor(label)

        return image
