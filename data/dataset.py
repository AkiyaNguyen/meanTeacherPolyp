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


def _resolve_depth_file(depth_dir: str, image_filename: str) -> str:
    """Resolve depth path by stem match; try .png before .jpg so RGB .jpg + depth .png works.

    If both ``stem.png`` and ``stem.jpg`` exist under ``depth_dir``, prefers ``.png`` (common for depth maps).
    Falls back to the exact ``image_filename`` under ``depth_dir`` last.
    """
    stem, _ext = os.path.splitext(image_filename)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        cand = os.path.join(depth_dir, stem + ext)
        print("check cand = ", cand)
        if os.path.isfile(cand):
            return cand
            print("return cand = ", cand)
            exit()
    primary = os.path.join(depth_dir, image_filename)
    return primary


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
            self.id_list.append(img_id.split('.')[0])
            self.img_list.append(os.path.join(self.data_path, image_dirname, img_id))  # Image paths
            self.gt_list.append(os.path.join(self.data_path, mask_dirname, img_id))  # Mask paths

        if require_depth:
            assert depth_dirname is not None, "depth_dirname is required if require_depth is True"
            depth_dir = os.path.join(self.data_path, depth_dirname)
            self.depth_list = []
            for img_id in self.images_list:
                self.depth_list.append(_resolve_depth_file(depth_dir, img_id))
            print("depth_list = ", self.depth_list)
            exit()


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
