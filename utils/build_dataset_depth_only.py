import os
import numpy as np
import torch
from data.transform import Resize, ToTensor
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
from test.eval import ImageFolderDataset
from torchvision import transforms


def build_dataset_depth_only(cfg):
    """Build dataset for depth-only training with npy files"""
    val_perc = int(cfg.get('data.val_split_perc', 0))
    resize_h = cfg.get('data.eval.resize_height', 320)
    resize_w = cfg.get('data.eval.resize_width', 320)
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor()
    ])

    train_dataloader, val_dataloader = None, None
    if val_perc == 0:
        train_data = dataset.DepthOnlyDataset(
            root=cfg.get('data.root'), 
            data2_dir=cfg.get('data.data2_dir'),
            mode='train', 
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=cfg.get('data.depth_dirname'), 
            list_name=None
        )
        train_num = len(train_data)
        print(f"Total training images: {train_num}")
        
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        
        print(f"Labelled images: {labeled_num}")
        print(f"Unlabelled images: {train_num - labeled_num}")
        
        batch_sampler = TwoStreamBatchSampler(
            train_num, labeled_num,
            int(cfg.get('data.labeled_bs')), 
            int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')),
            shuffle=cfg.get('data.shuffle', False)
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_sampler=batch_sampler,
            shuffle=False, num_workers=cfg.get('data.num_workers', 0)
        )

    elif val_perc < 100:
        print(f"Validation split: {val_perc}%")
        mask_dir = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), cfg.get('data.mask_dirname'))
        all_files = np.random.permutation([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]).tolist()
        total_num = len(all_files)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        print(f"Total training images: {train_num}, validation images: {val_num}")
        
        val_files = all_files[:val_num]
        train_files = all_files[val_num:]
        
        train_data = dataset.DepthOnlyDataset(
            root=cfg.get('data.root'), 
            data2_dir=cfg.get('data.data2_dir'),
            mode='train',
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=cfg.get('data.depth_dirname'),
            list_name=train_files
        )
        
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        
        print(f"Total training images: {train_num}, labelled: {labeled_num} ({labeled_num / train_num * 100:.2f}%)")
        
        batch_sampler = TwoStreamBatchSampler(
            train_num, labeled_num,
            int(cfg.get('data.labeled_bs')), 
            int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')),
            shuffle=cfg.get('data.shuffle', False)
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_sampler=batch_sampler,
            shuffle=False, num_workers=cfg.get('data.num_workers', 0)
        )
        
        # Create val dataloader (depth-only)
        val_data = DepthOnlyImageFolderDataset(
            dataset_root=os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir')),
            mask_dirname=cfg.get('data.test.mask_dirname'),
            depth_dirname=cfg.get('data.test.depth_dirname'),
            transform=val_test_transform, 
            list_name=val_files
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.get('data.test.batch_size', 4), shuffle=False, num_workers=0
        )
    else:
        raise ValueError("val_perc must be between 0 and 100.")

    # Test dataloader
    test_data = DepthOnlyImageFolderDataset(
        dataset_root=cfg.get('data.test.dataset_root'),
        mask_dirname=cfg.get('data.test.mask_dirname'),
        depth_dirname=cfg.get('data.test.depth_dirname'),
        transform=val_test_transform, 
        list_name=None
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.get('data.test.batch_size', 4), shuffle=False, num_workers=0
    )

    return train_dataloader, val_dataloader, test_dataloader


class DepthOnlyImageFolderDataset(torch.utils.data.Dataset):
    """Dataset for loading depth npy files and masks for evaluation"""
    def __init__(self, dataset_root: str, mask_dirname: str, depth_dirname: str, 
                 transform=None, list_name=None):
        self.dataset_root = dataset_root
        self.mask_dirname = mask_dirname
        self.depth_dirname = depth_dirname
        self.transform = transform
        
        # Get list of masks
        mask_dir = os.path.join(dataset_root, mask_dirname)
        if list_name is None:
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            mask_files = sorted(list_name)
        
        self.mask_files = mask_files
        self.id_list = [f.split('.')[0] for f in mask_files]

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, index):
        from PIL import Image
        
        mask_file = self.mask_files[index]
        img_id = self.id_list[index]
        
        # Load mask
        mask_path = os.path.join(self.dataset_root, self.mask_dirname, mask_file)
        mask = Image.open(mask_path).convert('L')
        
        # Load depth from npy
        depth_path = os.path.join(self.dataset_root, self.depth_dirname, img_id + '.npy')
        depth_array = np.load(depth_path)
        
        if len(depth_array.shape) == 3:
            depth_array = depth_array.squeeze(-1)
        
        # Normalize depth
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max > depth_min:
            depth_array = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_array = np.zeros_like(depth_array, dtype=np.uint8)
        
        depth = Image.fromarray(depth_array, mode='L')
        
        data = {'depth': depth, 'mask': mask}
        
        if self.transform:
            data = self.transform(data)
        
        return {**data, 'id': img_id}
