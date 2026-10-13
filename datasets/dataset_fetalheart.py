"""NPZ dataset and transforms for the fixed D10 fetal-heart split."""

from __future__ import annotations

import os
import random
import json

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _resize(image, label, output_size):
    height, width = image.shape
    if (height, width) == tuple(output_size):
        return image, label
    factors = (output_size[0] / height, output_size[1] / width)
    return zoom(image, factors, order=3), zoom(label, factors, order=0)


class FetalHeartTrainTransform:
    def __init__(self, output_size, augmentation="legacy"):
        self.output_size = output_size
        self.augmentation = augmentation

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.augmentation not in {"legacy", "interpolation_fixed", "none"}:
            raise ValueError(f"Unknown augmentation: {self.augmentation}")

        if self.augmentation != "none":
            if random.random() > 0.5:
                turns = np.random.randint(0, 4)
                axis = np.random.randint(0, 2)
                image = np.flip(np.rot90(image, turns), axis=axis).copy()
                label = np.flip(np.rot90(label, turns), axis=axis).copy()
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                image_order = 0 if self.augmentation == "legacy" else 3
                image = ndimage.rotate(image, angle, order=image_order, reshape=False, mode="reflect")
                label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")

        image, label = _resize(image, label, self.output_size)
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.int64)),
        }


class FetalHeartNPZDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, metadata_path=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        list_path = os.path.join(list_dir, f"{split}.txt")
        with open(list_path, encoding="utf-8") as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        self.metadata = {}
        if metadata_path is not None:
            with open(metadata_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.metadata = payload.get("cases", payload)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        case_name = self.sample_list[index]
        path = os.path.join(self.base_dir, case_name + ".npz")
        with np.load(path, allow_pickle=False) as data:
            image = np.asarray(data["image"])
            label = np.asarray(data["label"])
        image = np.squeeze(image)
        label = np.squeeze(label)
        if image.ndim != 2 or label.ndim != 2:
            raise ValueError(f"Expected 2-D image and label in {path}; got {image.shape}, {label.shape}")
        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case_name"] = case_name
        if self.metadata:
            if case_name not in self.metadata:
                raise KeyError(f"Missing view metadata for case '{case_name}'")
            sample["view_id"] = int(self.metadata[case_name]["view_id"])
        return sample
