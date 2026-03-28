import os
from typing import List
# from utils.transform import *
from .transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image


def _resolve_depth_file(depth_dir: str, img_id: str) -> str:
    """Pick depth file matching ``image_filename``; allow .png / .jpg / .jpeg interchangeably."""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        cand = os.path.join(depth_dir, img_id + ext)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Depth file not found for {img_id}")

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

# kvasir_SEG/ CVC-ClinicDB /kvasir_SEG + CVC-ClinicDB
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, require_depth=True, 
    list_name: List[str] | None = None, image_dirname: str = 'images', mask_dirname: str = 'masks', depth_dirname: str | None = 'depth-v1'):

        super(kvasir_SEG, self).__init__()
        self.data_path = os.path.join(root, data2_dir)
        self.require_depth = require_depth

        self.id_list = []
        self.img_list = []
        self.gt_list = []
        self.mode = mode

        self.images_list = os.listdir(os.path.join(self.data_path, image_dirname)) if list_name is None \
            else list_name
        self.images_list = sorted(self.images_list)

        for img_id in self.images_list:
            self.id_list.append(img_id.split('.')[0]) ## name without extension
            self.img_list.append(os.path.join(self.data_path, image_dirname, img_id))  # Image paths
            self.gt_list.append(os.path.join(self.data_path, mask_dirname, img_id))  # Mask paths

        if require_depth:
            assert depth_dirname is not None, "depth_dirname is required if require_depth is True"
            depth_dir = os.path.join(self.data_path, depth_dirname)
            self.depth_list = []
            for img_id in self.id_list:
                self.depth_list.append(_resolve_depth_file(depth_dir, img_id))


        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomRotation(degrees=90),
                    RandomZoom(zoom=(0.9, 1.1)),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    #ToTensor(),
                ])
        self.transform = transform


    def __getitem__(self, index, include_depth=None):
        include_depth = include_depth if include_depth is not None else self.require_depth
        assert int(include_depth) <= int(self.require_depth), \
            "avoid cases when dataset has no depth but depth is included in getitem"
        img = self.img_list[index]
        gt = self.gt_list[index]
        depth = self.depth_list[index] if include_depth else None

        img = Image.open(img).convert('RGB')
        gt = Image.open(gt).convert('L')
        # Load depth data wwith 3 channels for train mode
        if depth is not None:
            depth = Image.open(depth).convert('RGB') # type: ignore
        else:
            depth = None
            


        data = {'image': img, 'label': gt} if include_depth == False \
            else {'image': img, 'label': gt, 'depth': depth} # Add depth data in train mode

        if self.transform:
            data = self.transform(data)
            # data = ToTensor()(data)

            if self.mode == 'train':
                img_s1 = Image.fromarray(np.array((data['image'])).astype(np.uint8))
                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)
                data['image_s'] = img_s1 # type: ignore
        data = ToTensor()(data)
            
        return {**data, 'id': self.id_list[index]}

    def __len__(self):
        return len(self.img_list)


class DepthOnlyDataset(Dataset):
    """Dataset for training with raw depth (npy files) only, without RGB images"""
    def __init__(self, root, data2_dir, mode='train', transform=None, 
                 list_name: List[str] | None = None, mask_dirname: str = 'masks', 
                 depth_dirname: str = 'depth_npy'):
        
        super(DepthOnlyDataset, self).__init__()
        self.data_path = os.path.join(root, data2_dir)
        self.id_list = []
        self.gt_list = []
        self.depth_list = []
        self.mode = mode

        # Get mask files to determine the list of image IDs
        mask_dir = os.path.join(self.data_path, mask_dirname)
        mask_files = os.listdir(mask_dir) if list_name is None else list_name
        mask_files = sorted(mask_files)

        # Build file lists
        for mask_id in mask_files:
            img_id = mask_id.split('.')[0]  # Remove extension
            self.id_list.append(img_id)
            self.gt_list.append(os.path.join(self.data_path, mask_dirname, mask_id))
            
            # Resolve depth npy file
            depth_file = os.path.join(self.data_path, depth_dirname, img_id + '.npy')
            if not os.path.isfile(depth_file):
                raise FileNotFoundError(f"Depth npy file not found: {depth_file}")
            self.depth_list.append(depth_file)

        # Default transforms
        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomRotation(degrees=90),
                    RandomZoom(zoom=(0.9, 1.1)),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                ])
        self.transform = transform

    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        depth_path = self.depth_list[index]
        
        # Load mask
        gt = Image.open(gt_path).convert('L')
        
        # Load depth from npy and convert to PIL Image (grayscale)
        depth_array = np.load(depth_path)  # Shape: (H, W) or (H, W, 1)
        if len(depth_array.shape) == 3:
            depth_array = depth_array.squeeze(-1)
        
        # Normalize depth to 0-255 range for PIL
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max > depth_min:
            depth_array = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_array = np.zeros_like(depth_array, dtype=np.uint8)
        
        depth = Image.fromarray(depth_array, mode='L')
        
        # Create data dictionary with depth and mask
        data = {'depth': depth, 'label': gt}
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
            
            if self.mode == 'train':
                # Create augmented version
                depth_aug = Image.fromarray(np.array(data['depth']).astype(np.uint8), mode='L')
                if random.random() < 0.8:
                    depth_aug = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)(depth_aug)
                depth_aug = blur(depth_aug, p=0.5)
                data['depth_s'] = depth_aug  # type: ignore
        
        data = ToTensor()(data)
        return {**data, 'id': self.id_list[index]}
