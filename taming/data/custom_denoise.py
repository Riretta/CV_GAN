import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths_dn, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.target = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, target_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(target_images_list_file, "r") as f:
            paths_t = f.read().splitlines()
        self.data = ImagePaths_dn(paths=paths, path_target=paths_t, size=size, random_crop=False)




class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, test_target_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_target_images_list_file, "r") as f:
            paths_t = f.read().splitlines()
        self.data = ImagePaths_dn(paths=paths, path_target=paths_t,size=size, random_crop=False)
