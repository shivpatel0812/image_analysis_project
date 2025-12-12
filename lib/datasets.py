import os
import numpy as np
from PIL import Image, ImageChops

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

from tqdm import tqdm

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def load_medical_image(path):
    if path.endswith('.mhd') or path.endswith('.nii') or path.endswith('.nii.gz'):
        if not HAS_SITK:
            raise RuntimeError("SimpleITK required for .mhd files: pip install SimpleITK")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8))
    else:
        return Image.open(path)


class ImageListDataset(VisionDataset):
    def __init__(self, data_root, listfile, transform, gray=False, nolabel=False, multiclass=False):
        self.image_list = []
        self.label_list = []
        self.nolabel = nolabel
        self.data_root = data_root
        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split()
                image_path = os.path.join(data_root, items[0])
                if not nolabel:
                    if not multiclass:
                        label = int(items[1])
                    elif multiclass:
                        label = list(map(float, items[1:]))
                    else:
                        raise ValueError("Line format is not right")
                self.image_list.append(image_path)
                if not nolabel:
                    self.label_list.append(label)

        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        
        if img_path.endswith('.mhd') or img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
            image = load_medical_image(img_path)
        else:
            image = Image.open(img_path)
        
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = self.transform(image)

        if not self.nolabel:
            label = self.label_list[index]
            return image, torch.tensor(label)
        else:
            return image
