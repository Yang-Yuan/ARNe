import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob


class PGM(Dataset):

    def __init__(self, split = "train", dir_path = "."):
        self.filenames = [f.replace("\\", "/") for f in glob(os.path.join(dir_path, "*", "*.npz")) if split in f]
        print(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        path = self.filenames[index]
        data = np.load(path)
        image = data.get("image").reshape(16, 160, 160)
        image = torch.from_numpy(image)
        target = data.get("target")
        target = torch.from_numpy(target)
        meta_target = data.get("meta_target")
        meta_target = torch.from_numpy(meta_target)
        return image, target, meta_target, index + 1

    @staticmethod
    def cast_data(images, target, meta_target):
        target = target.long()
        meta_target = meta_target.float()

        images = images.float()

        images_context = images[:, :8, :, :, ]
        images_choices = images[:, 8:, :, :, ]

        return images_context, images_choices, target, meta_target
