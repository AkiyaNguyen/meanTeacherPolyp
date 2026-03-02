from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch

eval_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = pred.round().float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = gt.round().float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.tensor(1.0, device=pred.device)
    # Acc
    ACC_overall = (TP + TN) / (TP + FP + FN + TN + 1e-8)


    # IoU
    IoU_poly = TP / (TP + FP + FN + 1e-8)

    # Dice
    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = gt.view(size, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_score = torch.mean((2 * intersection + 1e-8) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8))

    return {
        'ACC_overall': ACC_overall.item(),
        'Dice': dice_score.item(),
        'IoU': IoU_poly.item(),
    }


class ImageFolderDataset(Dataset):
    """Dataset for loading image and mask pairs from folders."""
    def __init__(self, dataset_root, image_dirname, mask_dirname, transform=None):
        self.image_path = os.path.join(dataset_root, image_dirname)
        self.mask_path = os.path.join(dataset_root, mask_dirname)
        self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
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

        if self.transform:
            data = {'image': img, 'label': mask}
            data = self.transform(data)
            img_resized, mask_resized = data['image'], data['label']
        else:
            img_resized = img
            mask_resized = mask

        return {
            'image': img_resized,
            'mask': mask_resized,
            'filename': self.image_files[index],
            'original_size': original_size
        }
