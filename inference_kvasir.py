"""
Kvasir inference — Mean Teacher (Student + Teacher RGB / RGB-D).

Run inference on the Kvasir test set using trained student and teacher checkpoints.
Saves segmentation masks to out_dir/stu, out_dir/tea_rgb, out_dir/tea_rgbd.
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from engine.Config import Config
from utils.common import get_proper_device
from test.eval import ImageFolderDataset
from data.transform import Resize, ToTensor
import models


def parse_args():
    p = argparse.ArgumentParser(
        description="Kvasir inference: save stu, tea_rgb, tea_rgbd masks"
    )
    p.add_argument(
        "--stu_ckpt",
        type=str,
        default="save_dir/depth_enhance_mt_epoch105.pth",
        help="Path to student checkpoint (.pth)",
    )
    p.add_argument(
        "--tea_ckpt",
        type=str,
        default="save_dir/depth_enhance_mt_teacher_epoch105.pth",
        help="Path to teacher checkpoint (.pth)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="kvasir_inference",
        help="Directory to save predicted masks (subdirs: stu, tea_rgb, tea_rgbd)",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="../polypdepth_dataset/TestDataset/Kvasir",
        help="Kvasir test dataset root (images, masks, depth-v1 dirs)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="cfg/depth_enhance_mt.yaml",
        help="Config YAML for model names and test data paths",
    )
    p.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    p.add_argument("--device", type=str, default=None, help="Device (default: from config)")
    return p.parse_args()


def save_mask(tensor, path, original_size=None):
    """
    tensor: (H,W) or (1,H,W) in [0,1]; save as 0/255 PNG.
    original_size: (width, height) of original image; if set, mask is resized to this size
    so the saved image matches original resolution (no pixelation in draw.io / figures).
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # (1,1,H,W)
    if original_size is not None:
        # original_size is (width, height); interpolate expects (N,C,H,W) with H,W = height, width
        h, w = original_size[1], original_size[0]
        tensor = torch.nn.functional.interpolate(
            tensor, size=(h, w), mode="nearest"
        )
    tensor = tensor.squeeze(0).squeeze(0)
    arr = (tensor.cpu().numpy() >= 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def main():
    args = parse_args()

    cfg = Config(config_file=args.config)
    device = get_proper_device(args.device or cfg.get("device"))
    num_classes = cfg.get("model.num_channels_output", 1)
    dataset_root = args.dataset_root
    resize_h = cfg.get("data.test.resize_height", 320)
    resize_w = cfg.get("data.test.resize_width", 320)
    image_dir = cfg.get("data.test.image_dirname", "images")
    mask_dir = cfg.get("data.test.mask_dirname", "masks")
    depth_dir = str(cfg.get("data.test.depth_dirname", "depth-v1"))

    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor(),
    ])
    eval_data = ImageFolderDataset(
        dataset_root=dataset_root,
        image_dirname=image_dir,
        mask_dirname=mask_dir,
        depth_dirname=depth_dir,
        transform=val_test_transform,
        list_name=None,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Dataset root: {dataset_root} | samples: {len(eval_data)}")

    stu_model = getattr(models, cfg.get("model.stu_model.name"))(
        num_classes=num_classes
    ).to(device)
    tea_model = getattr(models, cfg.get("model.tea_model.name"))(
        num_classes=num_classes
    ).to(device)

    stu_model.load_state_dict(
        torch.load(args.stu_ckpt, map_location=device), strict=True
    )
    tea_model.load_state_dict(
        torch.load(args.tea_ckpt, map_location=device), strict=True
    )
    stu_model.eval()
    tea_model.eval()
    print("Models loaded.")

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ("stu", "tea_rgb", "tea_rgbd"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    with torch.no_grad():
        for batch in eval_loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            filenames = batch["filename"]
            original_sizes = batch["original_size"]
            if isinstance(filenames, str):
                filenames = [filenames]
            else:
                filenames = list(filenames)
            # original_size from dataset is (width, height); batch may be single tuple or list of tuples
            if isinstance(original_sizes, (list, tuple)) and len(original_sizes) > 0 and isinstance(original_sizes[0], (list, tuple)):
                original_sizes = list(original_sizes)
            else:
                original_sizes = [original_sizes] * len(filenames)

            stu_out = stu_model(img)
            tea_out = tea_model(img, depth)
            tea_rgb = tea_out["rgb"]
            tea_rgbd = tea_out["rgb_depth"]

            for i in range(img.size(0)):
                base = os.path.splitext(filenames[i])[0]
                orig_size = original_sizes[i]  # (width, height)
                save_mask(
                    stu_out[i],
                    os.path.join(args.out_dir, "stu", base + ".png"),
                    original_size=orig_size,
                )
                save_mask(
                    tea_rgb[i],
                    os.path.join(args.out_dir, "tea_rgb", base + ".png"),
                    original_size=orig_size,
                )
                save_mask(
                    tea_rgbd[i],
                    os.path.join(args.out_dir, "tea_rgbd", base + ".png"),
                    original_size=orig_size,
                )

    print(
        f"Done. Masks saved under {args.out_dir}/stu, "
        f"{args.out_dir}/tea_rgb, {args.out_dir}/tea_rgbd"
    )


if __name__ == "__main__":
    main()
