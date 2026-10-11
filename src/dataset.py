"""
PyTorch Dataset / DataLoader construction from the split manifest produced by
split_patient_level.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocessing import get_train_transforms, get_eval_transforms, compute_class_weights


class BloodSmearDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, transforms=None):
        self.df = manifest_df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = np.array(img)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        label = int(row["label"])
        subject_id = row["subject_id"]
        return img, label, subject_id


def load_manifest(manifest_path: str) -> pd.DataFrame:
    return pd.read_csv(manifest_path)


def build_dataloaders(cfg: dict, use_balanced_sampler: bool = True):
    manifest = load_manifest(cfg["data"]["split_manifest"])
    image_size = cfg["data"]["image_size"]

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]

    train_ds = BloodSmearDataset(train_df, transforms=get_train_transforms(image_size))
    val_ds = BloodSmearDataset(val_df, transforms=get_eval_transforms(image_size))
    test_ds = BloodSmearDataset(test_df, transforms=get_eval_transforms(image_size))

    bs = cfg["dataloader"]["batch_size"]
    nw = cfg["dataloader"]["num_workers"]
    pm = cfg["dataloader"]["pin_memory"]

    if use_balanced_sampler:
        w0, w1 = compute_class_weights(train_df["label"].values)
        sample_weights = train_df["label"].map({0: w0, 1: w1}).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                   num_workers=nw, pin_memory=pm)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                   num_workers=nw, pin_memory=pm)

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    return train_loader, val_loader, test_loader
