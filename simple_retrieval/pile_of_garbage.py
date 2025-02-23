
import torch
from time import time
from tqdm import tqdm
import numpy as np
import os
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import os
from typing import List, Tuple, Callable, Optional
from PIL import Image
import h5py


class CustomImageFolder(Dataset):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        """
        Args:
            root (str): Root directory path.
            transform (Callable, optional): A function/transform to apply to the images.
            extensions (tuple): Tuple of allowed file extensions (default: common image formats).
        """
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.samples = self._make_dataset()

    def _is_valid_file(self, filename: str) -> bool:
        """Checks if a file is a valid image file based on its extension."""
        return filename.lower().endswith(self.extensions)

    def _make_dataset(self) -> List[str]:
        """Indexes all valid image files in the directory and its subdirectories."""
        images = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if self._is_valid_file(file):
                    images.append(os.path.join(root, file))
        return sorted(images)

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Returns the image and its file path at the given index."""
        filepath = self.samples[index]
        with Image.open(filepath) as img:
            img = img.convert("RGB")  # Ensure all images are in RGB format
            if self.transform:
                img = self.transform(img)
        return img, filepath

class CustomImageFolderFromFileList(CustomImageFolder):
    def __init__(self,
                 file_list: List[str] = None,
                 transform: Optional[Callable] = None ):
        """
        Args:
            root (str): Root directory path.
            transform (Callable, optional): A function/transform to apply to the images.
            extensions (tuple): Tuple of allowed file extensions (default: common image formats).
        """
        self.transform = transform
        self.samples = file_list
    def _is_valid_file(self, filename: str) -> bool:
        """Checks if a file is a valid image file based on its extension."""
        return filename.lower().endswith(self.extensions)

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Returns the image and its file path at the given index."""
        filepath = self.samples[index]
        with Image.open(filepath) as img:
            img = img.convert("RGB")  # Ensure all images are in RGB format
            w, h = img.size
            if self.transform:
                img = self.transform(img)
        return img, h, w, filepath

def collate_with_string(batch):
    """
    Custom collate function for a dataset where each item is a tuple of
    (image_tensor, label, string).

    Args:
        batch: List of tuples (image_tensor, label, string).

    Returns:
        Tuple:
        - Batch of image tensors (stacked into a single tensor).
        - Batch of labels (as a tensor or list).
        - Batch of strings (as a list).
    """
    # Unpack the batch into separate lists
    images, heights, widths, strings = zip(*batch)

    # Stack image tensors into a batch (B, C, H, W)
    image_batch = torch.stack(images)
    heights_batch = heights
    widths_batch = widths
    # Strings are kept as a list
    string_batch = strings

    return image_batch, heights_batch, widths_batch, string_batch

def no_collate(batch):
    # Unpack the batch into separate lists
    return  zip(*batch)



class H5LocalFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, fnames_list=[]):
        self.dir_path = dir_path
        self.descriptors_dataset = None
        self.lafs_dataset = None
        self.hw_dataset = None
        self.fname_list = fnames_list
        self.dataset_len = len(fnames_list)

    def __getitem__(self, index):
        key = self.fname_list[index]
        if self.descriptors_dataset is None:
            self.descriptors_dataset = h5py.File(os.path.join(self.dir_path, "descriptors.h5"), 'r')
            self.lafs_dataset = h5py.File(os.path.join(self.dir_path, "lafs.h5"), 'r')
            self.hw_dataset = h5py.File(os.path.join(self.dir_path, "hw.h5"), 'r')
        return torch.from_numpy(self.descriptors_dataset[key][...]), torch.from_numpy(self.lafs_dataset[key][...]), torch.from_numpy(self.hw_dataset[key][...]), key

    def __len__(self):
        return self.dataset_len
