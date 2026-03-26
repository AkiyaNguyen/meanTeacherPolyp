from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
from typing import List
from data.dataset import _resolve_depth_file
# from ..data.transform import *

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    # Binarize: pred in [0,1] from sigmoid; threshold 0.5
    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    # GT: support both 0/255 (ToTensor -> 0 and 1) and 0/1 (ToTensor -> 0 and 1/255)
    gt_max = gt.max().item()
    gt_binary = (gt > 0.5).float() if gt_max > 0.5 else (gt > 0).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    # Acc
    ACC_overall = (TP + TN) / (TP + FP + FN + TN + 1e-8)
    # IoU (no overlap -> 0)
    IoU_poly = TP / (TP + FP + FN + 1e-8)

    # Dice (same binarization as ACC/IoU)
    size = pred_binary.size(0)
    pred_flat = pred_binary.view(size, -1)
    target_flat = gt_binary.view(size, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_score = torch.mean((2 * intersection + 1e-8) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8))

    return {
        'ACC_overall': ACC_overall.item(),
        'Dice': dice_score.item(),
        'IoU': IoU_poly.item(),
    }


class ImageFolderDataset(Dataset):
    """Dataset for loading image and mask pairs from folders."""
    def __init__(self, dataset_root, image_dirname, mask_dirname, depth_dirname='',transform=None, list_name: List[str] | None = None):
        self.image_path = os.path.join(dataset_root, image_dirname)
        self.mask_path = os.path.join(dataset_root, mask_dirname)
        if depth_dirname != '' and depth_dirname is not None:    
            self.depth_path = os.path.join(dataset_root, depth_dirname)
        else:
            self.depth_path = None

        self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]) if \
            list_name is None else list_name
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_path, self.image_files[index])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size

        mask_name = self.image_files[index]
        mask_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(mask_path).convert('L')

        depth = None
        if self.depth_path is not None:
            depth_name_without_extension = self.image_files[index].split('.')[0]
            depth_path = _resolve_depth_file(self.depth_path, depth_name_without_extension)
            depth = Image.open(depth_path).convert('RGB')


        if self.transform:
            data = {'image': img, 'mask': mask}
            if depth is not None:
                data.update({'depth': depth})
            
            data = self.transform(data)
        return {
            **data,
            'filename': self.image_files[index],
            'original_size': original_size
        }
