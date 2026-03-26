import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import models
from test.eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load student/teacher from MT checkpoint, compare Dice, and visualize ranked results."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="save_dir/best_85_dice_emaEncoderOnlyTrain.pth",
        help="Path to checkpoint containing stu_model and tea_model.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"D:\HCMUS_ComputerScience\selfStudy\DeepLearningStuff\Paper\PolypSegmentation\code\polypdepth_dataset\TestDataset\Kvasir",
        help="Dataset root containing images/masks/depth-v1 folders.",
    )
    parser.add_argument("--image_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="masks")
    parser.add_argument("--depth_dirname", type=str, default="depth-v1")
    parser.add_argument("--stu_model_name", type=str, default="ResNet34U_f")
    parser.add_argument("--tea_model_name", type=str, default="DepthFusion_ResNet34U_f_EMAEncoderOnly")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--resize_h", type=int, default=320)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rows_per_page", type=int, default=20)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="analysis/stu_tea_compare")
    parser.add_argument(
        "--sort_desc",
        action="store_true",
        help="Sort by (stu_dice - tea_dice) descending. Default is ascending.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively. By default figures are only saved.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_models(args: argparse.Namespace, device: torch.device):
    stu_model = getattr(models, args.stu_model_name)(num_classes=args.num_classes).to(device)
    tea_model = getattr(models, args.tea_model_name)(num_classes=args.num_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "stu_model" not in ckpt or "tea_model" not in ckpt:
        raise KeyError("Checkpoint must contain 'stu_model' and 'tea_model' keys.")

    stu_model.load_state_dict(ckpt["stu_model"])

    tea_state = ckpt["tea_model"]
    try:
        tea_model.load_state_dict(tea_state)
    except RuntimeError:
        # Backward-compatibility for SE block naming drift:
        # old checkpoints: fusion_blockX.se.{...}
        # current code:    fusion_blockX.se.se.{...}
        remapped = {}
        for k, v in tea_state.items():
            if ".fusion_block" in f".{k}" and ".se." in k and ".se.se." not in k:
                remapped[k.replace(".se.", ".se.se.", 1)] = v
            else:
                remapped[k] = v
        tea_model.load_state_dict(remapped)

    stu_model.eval()
    tea_model.eval()
    return stu_model, tea_model


def build_sample_list(args: argparse.Namespace) -> List[Dict[str, str]]:
    image_dir = os.path.join(args.dataset_root, args.image_dirname)
    mask_dir = os.path.join(args.dataset_root, args.mask_dirname)
    depth_dir = os.path.join(args.dataset_root, args.depth_dirname)

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    samples: List[Dict[str, str]] = []
    for fn in image_files:
        sample = {
            "filename": fn,
            "image": os.path.join(image_dir, fn),
            "mask": os.path.join(mask_dir, fn),
            "depth": os.path.join(depth_dir, fn),
        }
        if not (os.path.exists(sample["mask"]) and os.path.exists(sample["depth"])):
            continue
        samples.append(sample)
    return samples


def run_inference_ranked(
    stu_model: torch.nn.Module,
    tea_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict]:
    tf = transforms.Compose(
        [
            transforms.Resize((args.resize_h, args.resize_w)),
            transforms.ToTensor(),
        ]
    )

    ranked: List[Dict] = []
    samples = build_sample_list(args)

    with torch.no_grad():
        for sample in samples:
            image_pil = Image.open(sample["image"]).convert("RGB")
            mask_pil = Image.open(sample["mask"]).convert("L")
            depth_pil = Image.open(sample["depth"]).convert("RGB")

            image_t = tf(image_pil).unsqueeze(0).to(device)
            mask_t = tf(mask_pil).unsqueeze(0).to(device)
            depth_t = tf(depth_pil).unsqueeze(0).to(device)

            stu_out = stu_model(image_t)
            tea_out = tea_model(image_t, depth_t)

            stu_dice = evaluate(stu_out, mask_t)["Dice"]
            tea_dice = evaluate(tea_out, mask_t)["Dice"]

            # Visualize probability maps (continuous [0, 1]) instead of thresholded masks.
            stu_mask = stu_out.float().squeeze(0)
            tea_mask = tea_out.float().squeeze(0)
            gt_mask = (mask_t > 0.5).float().squeeze(0)

            ranked.append(
                {
                    "filename": sample["filename"],
                    "stu_dice": float(stu_dice),
                    "tea_dice": float(tea_dice),
                    "dice_diff": float(stu_dice - tea_dice),
                    "image": _to_numpy_image(image_t.squeeze(0)),
                    "stu_mask": _to_numpy_mask(stu_mask),
                    "tea_mask": _to_numpy_mask(tea_mask),
                    "gt_mask": _to_numpy_mask(gt_mask),
                    "depth": _to_numpy_image(depth_t.squeeze(0)),
                }
            )

    ranked.sort(key=lambda x: x["dice_diff"], reverse=args.sort_desc)
    return ranked


def save_json_order(ranked: List[Dict], output_dir: str, sort_desc: bool) -> str:
    order_data = {
        "sort_key": "student_dice_minus_teacher_dice",
        "sort_desc": sort_desc,
        "total_images": len(ranked),
        "images": [
            {
                "rank": i + 1,
                "filename": item["filename"],
                "student_dice": item["stu_dice"],
                "teacher_dice": item["tea_dice"],
                "dice_diff": item["dice_diff"],
            }
            for i, item in enumerate(ranked)
        ],
    }
    json_path = os.path.join(output_dir, "display_order.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(order_data, f, indent=2)
    return json_path


def plot_ranked_pages(ranked: List[Dict], output_dir: str, rows_per_page: int, show: bool):
    cols = 5  # image | stu | tea | gt | depth
    page_count = (len(ranked) + rows_per_page - 1) // rows_per_page

    for page_idx in range(page_count):
        start = page_idx * rows_per_page
        end = min(len(ranked), (page_idx + 1) * rows_per_page)
        chunk = ranked[start:end]

        fig_h = max(4, len(chunk) * 2.4)
        fig, axes = plt.subplots(len(chunk), cols, figsize=(16, fig_h), squeeze=False)
        if len(chunk) > 0:
            axes[0, 0].set_title("Image")
            axes[0, 1].set_title("Student Mask")
            axes[0, 2].set_title("Teacher Mask")
            axes[0, 3].set_title("Ground Truth")
            axes[0, 4].set_title("Depth-v1")

        for row_i, item in enumerate(chunk):
            axes[row_i, 0].imshow(item["image"])
            axes[row_i, 1].imshow(item["stu_mask"], cmap="gray", vmin=0, vmax=1)
            axes[row_i, 2].imshow(item["tea_mask"], cmap="gray", vmin=0, vmax=1)
            axes[row_i, 3].imshow(item["gt_mask"], cmap="gray", vmin=0, vmax=1)
            axes[row_i, 4].imshow(item["depth"])

            label = (
                f"#{start + row_i + 1}"
            )
            axes[row_i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, labelpad=110)
            axes[row_i, 0].set_title(f"{item['filename']}", fontsize=8)
            axes[row_i, 1].set_title(f"Stu Dice={item['stu_dice']:.4f}", fontsize=8)
            axes[row_i, 2].set_title(f"Tea Dice={item['tea_dice']:.4f}", fontsize=8)
            axes[row_i, 3].set_title("GT", fontsize=8)
            axes[row_i, 4].set_title(f"Depth | Diff={item['dice_diff']:.4f}", fontsize=8)

            for c in range(cols):
                axes[row_i, c].axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"ranked_page_{page_idx + 1:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    stu_model, tea_model = load_models(args, device)
    ranked = run_inference_ranked(stu_model, tea_model, args, device)

    if len(ranked) == 0:
        raise RuntimeError("No valid images found to process.")

    json_path = save_json_order(ranked, args.output_dir, args.sort_desc)
    plot_ranked_pages(ranked, args.output_dir, args.rows_per_page, args.show)

    print(f"Done. Processed {len(ranked)} images.")
    print(f"JSON order saved to: {json_path}")
    print(f"Figure pages saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


# python infer_compare_student_teacher.py   --checkpoint "save_dir/best_85_dice_emaEncoderOnlyTrain.pth"   
# --dataset_root "D:/HCMUS_ComputerScience/selfStudy/DeepLearningStuff/Paper/PolypSegmentation/code/polypdepth_dataset/TestDataset/Kvasir"   --stu_model_name "ResNet34U_f"   --tea_model_name "DepthFusion_ResNet34U_f_EMAEncoderOnly"   --rows_per_page 20   --output_dir "analysis/stu_tea_compare"   --sort_desc