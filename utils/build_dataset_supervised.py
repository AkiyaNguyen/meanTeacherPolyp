"""
Train loaders for fully supervised runs (no TwoStreamBatchSampler).

Uses the same labeled count rules as ``build_dataset`` (percentage / labeled_num, even adjustment)
and the same index range as the primary stream in ``TwoStreamBatchSampler(..., shuffle=False)``:
indices ``0 .. labeled_num - 1``.
"""

import os
import numpy as np
import torch
from torch.utils.data import Subset
from data.transform import Resize, ToTensor
from data import dataset
from test.eval import ImageFolderDataset
from torchvision import transforms


def _labeled_num(train_num, cfg):
    if cfg.get('data.label_mode') == 'percentage':
        n = round(train_num * cfg.get('data.labeled_perc') / 100)
        if n % 2 != 0:
            n -= 1
    else:
        n = cfg.get('data.labeled_num')
    return n


def build_dataset_supervised(cfg):
    """
    Like ``utils.build_dataset.build_dataset`` but train DataLoader is
    ``Subset(train_data, range(labeled_num))`` with normal batching (all samples labeled).
    Val/test construction matches ``build_dataset``.
    """
    val_perc = int(cfg.get('data.val_split_perc', 0))
    resize_h = cfg.get('data.eval.resize_height', 320)
    resize_w = cfg.get('data.eval.resize_width', 320)
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor()
    ])

    train_dataloader, val_dataloader = None, None
    if val_perc == 0:
        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth'), list_name=None,
            image_dirname=cfg.get('data.image_dirname'),
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=cfg.get('data.depth_dirname', None),
            )
        train_num = len(train_data)
        print(f"[supervised] Total training images: {train_num}")
        labeled_num = _labeled_num(train_num, cfg)
        print(f"[supervised] Labeled subset for training: {labeled_num} (rest unused)")
        train_dataloader = torch.utils.data.DataLoader(
            Subset(train_data, list(range(labeled_num))),
            batch_size=int(cfg.get('data.batch_size')),
            shuffle=bool(cfg.get('data.shuffle')),
            num_workers=cfg.get('data.num_workers'),
        )

    elif val_perc < 100:
        print(f"[supervised] Validation split: {val_perc}%")
        data_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), 'images')
        all_files = np.random.permutation([f for f in os.listdir(data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]).tolist()
        total_num = len(all_files)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        print(f"[supervised] Total training images: {train_num}, validation images: {val_num}")
        val_files = all_files[:val_num]
        train_files = all_files[val_num:]
        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth'), 
            list_name=train_files,
            image_dirname=cfg.get('data.image_dirname'),
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=cfg.get('data.depth_dirname', None),
            )
        labeled_num = _labeled_num(train_num, cfg)
        print(f"[supervised] Labeled subset: {labeled_num} ({labeled_num / train_num * 100:.2f}% of train split)")
        train_dataloader = torch.utils.data.DataLoader(
            Subset(train_data, list(range(labeled_num))),
            batch_size=int(cfg.get('data.batch_size')),
            shuffle=bool(cfg.get('data.shuffle')),
            num_workers=cfg.get('data.num_workers'),
        )
        train_dataset_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'))
        val_data = ImageFolderDataset(
            dataset_root=train_dataset_root,
            image_dirname=cfg.get('data.test.image_dirname'), 
            mask_dirname=cfg.get('data.test.mask_dirname'), 
            depth_dirname=cfg.get('data.test.depth_dirname', None),
            transform=val_test_transform, list_name=val_files)
        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.get('data.test.batch_size'), shuffle=False, num_workers=0)
    else:
        raise ValueError("val_perc must be between 0 and 100.")

    test_data = ImageFolderDataset(
        dataset_root=cfg.get('data.test.dataset_root'),
        image_dirname=cfg.get('data.test.image_dirname'),
        mask_dirname=cfg.get('data.test.mask_dirname'),
        depth_dirname=cfg.get('data.test.depth_dirname', None),
        transform=val_test_transform, list_name=None)

    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.get('data.test.batch_size'), shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
