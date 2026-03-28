import os
import numpy as np
import torch
from data.transform import Resize, ToTensor
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
from test.eval import ImageFolderDataset
from torchvision import transforms


def build_dataset_image_as_depth(cfg):
    """
    Build dataloaders for image-as-depth training mode.

    This mode never loads depth from disk. The RGB image is reused as the
    depth-branch input inside the trainer/model forward pass.
    """
    val_perc = int(cfg.get('data.val_split_perc', 0))
    resize_h = cfg.get('data.test.resize_height', 320)
    resize_w = cfg.get('data.test.resize_width', 320)
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor(),
    ])

    train_dataloader, val_dataloader = None, None

    if val_perc == 0:
        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'),
            data2_dir=cfg.get('data.data2_dir'),
            mode='train',
            require_depth=False,
            image_dirname=cfg.get('data.image_dirname'),
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=None,
            list_name=None,
        )
        train_num = len(train_data)
        print(f'Total training images: {train_num}')

        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')

        print(f'Labelled images: {labeled_num}')
        print(f'Unlabelled images: {train_num - labeled_num}')

        batch_sampler = TwoStreamBatchSampler(
            train_num,
            labeled_num,
            int(cfg.get('data.labeled_bs')),
            int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')),
            shuffle=cfg.get('data.shuffle', False),
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            shuffle=False,
            num_workers=cfg.get('data.num_workers', 0),
        )

    elif val_perc < 100:
        print(f'Validation split: {val_perc}%')
        image_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), cfg.get('data.image_dirname'))
        all_files = np.random.permutation(
            [f for f in os.listdir(image_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        ).tolist()

        total_num = len(all_files)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        print(f'Total training images: {train_num}, validation images: {val_num}')

        val_files = all_files[:val_num]
        train_files = all_files[val_num:]

        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'),
            data2_dir=cfg.get('data.data2_dir'),
            mode='train',
            require_depth=False,
            image_dirname=cfg.get('data.image_dirname'),
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=None,
            list_name=train_files,
        )

        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')

        print(f'Total training images: {train_num}, labelled: {labeled_num} ({labeled_num / train_num * 100:.2f}%)')

        batch_sampler = TwoStreamBatchSampler(
            train_num,
            labeled_num,
            int(cfg.get('data.labeled_bs')),
            int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')),
            shuffle=cfg.get('data.shuffle', False),
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            shuffle=False,
            num_workers=cfg.get('data.num_workers', 0),
        )

        train_dataset_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'))
        val_data = ImageFolderDataset(
            dataset_root=train_dataset_root,
            image_dirname=cfg.get('data.test.image_dirname'),
            mask_dirname=cfg.get('data.test.mask_dirname'),
            depth_dirname='',
            transform=val_test_transform,
            list_name=val_files,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=cfg.get('data.test.batch_size', 1),
            shuffle=False,
            num_workers=0,
        )
    else:
        raise ValueError('val_perc must be between 0 and 100.')

    test_data = ImageFolderDataset(
        dataset_root=cfg.get('data.test.dataset_root'),
        image_dirname=cfg.get('data.test.image_dirname'),
        mask_dirname=cfg.get('data.test.mask_dirname'),
        depth_dirname='',
        transform=val_test_transform,
        list_name=None,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.get('data.test.batch_size', 1),
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, test_dataloader
