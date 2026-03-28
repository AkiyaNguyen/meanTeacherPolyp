# Hướng Dẫn Training với Raw Depth (Depth-Only)

## Tổng Quan
Bạn có thể giờ đây train mô hình chỉ bằng raw depth từ các file `.npy` mà không cần RGB images. Điều này hữu ích khi bạn có sẵn dữ liệu depth raw dưới dạng numpy arrays.

## Cấu Trúc Dữ Liệu Yêu Cầu

```
dataset_root/
├── TrainDataset/
│   ├── masks/              # Mask files (PNG/JPG)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── depth_npy/          # Raw depth files (numpy arrays)
│       ├── image1.npy
│       ├── image2.npy
│       └── ...
└── TestDataset/
    └── Kvasir/
        ├── masks/
        │   ├── test1.png
        │   ├── test2.png
        │   └── ...
        └── depth_npy/
            ├── test1.npy
            ├── test2.npy
            └── ...
```

## Định Dạng File `.npy`

- **Shape**: `(H, W)` hoặc `(H, W, 1)` - Depth map dưới dạng 2D hoặc 3D array
- **Dtype**: Bất kỳ numeric type nào (float32, float64, uint16, etc.)
- **Normalization**: Tự động normalize từ [min, max] → [0, 255]

Ví dụ tạo file `.npy`:
```python
import numpy as np

# Giả sử depth là một numpy array với shape (H, W)
depth_array = ...  # shape: (320, 320), values: [0, 4095] (16-bit depth)
np.save('image1.npy', depth_array)
```

## Config File

File config: `cfg/depthOnly.yaml`

**Các thay đổi chính so với `emaEncoderOnly.yaml`:**

```yaml
model:
  stu_model:
    name: DepthOnlyResNet34U_f           # ← Model độc lập với depth
  tea_model:
    name: DepthOnlyResNet34U_f_EMAEncoderOnly  # ← Teacher model cho EMA

data:
  dataset: DepthOnlyDataset              # ← Dataset mới
  depth_dirname: depth_npy               # ← Folder chứa .npy files
  # Không cần image_dirname hoặc require_depth
```

## Các File Được Thêm/Sửa

### 1. **models/ResUNet.py** (Thêm)
- `DepthOnlyEncoder`: Encoder chuyên xử lý 1-channel depth input
- `DepthOnlyResNet34U_f`: Model baseline cho depth-only training
- `DepthOnlyResNet34U_f_EMAEncoderOnly`: Model cho semi-supervised learning với EMA

### 2. **data/dataset.py** (Thêm)
- `DepthOnlyDataset`: Dataset class load depth từ `.npy` files
  - Tự động normalize depth về [0, 255]
  - Hỗ trợ data augmentation (flip, rotate, zoom, blur)
  - Return: `{'depth_s': tensor (1xHxW), 'depth': tensor (1xHxW), 'label': tensor (1xHxW), 'id': str}`

### 3. **utils/build_dataset_depth_only.py** (Tạo mới)
- `build_dataset_depth_only()`: Build train/val/test dataloaders
- `DepthOnlyImageFolderDataset`: Dataset cho evaluation

### 4. **depthOnlyTrain.py** (Tạo mới)
- `DepthOnly_MT_Trainer_EMAEncoderOnly`: Training loop cho depth-only
  - Phase 1: Train student + EMA encoder
  - Phase 2: Train teacher decoder, freeze encoder
- `DepthOnlyEvalHook_EMAEncoderOnly`: Evaluation hook

### 5. **cfg/depthOnly.yaml** (Tạo mới)
- Config file mẫu cho depth-only training

## Cách Sử Dụng

### 1. **Chuẩn bị dữ liệu:**
```bash
# Đảm bảo folder structure đúng:
# ../polypdepth_dataset/TrainDataset/masks/
# ../polypdepth_dataset/TrainDataset/depth_npy/
# ../polypdepth_dataset/TestDataset/Kvasir/masks/
# ../polypdepth_dataset/TestDataset/Kvasir/depth_npy/
```

### 2. **Chạy training:**
```bash
# Training cơ bản
python depthOnlyTrain.py --config cfg/depthOnly.yaml

# Với hyperparameter override
python depthOnlyTrain.py --config cfg/depthOnly.yaml data.labeled_perc 20

# Với Optuna hyperparameter tuning (4 trials)
python depthOnlyTrain.py --optuna_trial_times 4 --config cfg/depthOnly.yaml

# Load checkpoint
python depthOnlyTrain.py --config cfg/depthOnly.yaml Trainer.load_ckpt_path save_dir/checkpoint.pth
```

## Các Mô Hình Có Sẵn

| Model | Mục Đích | Input |
|-------|---------|-------|
| `DepthOnlyResNet34U_f` | Supervised learning dùng làm student | (B, 1, H, W) |
| `DepthOnlyResNet34U_f_EMAEncoderOnly` | Semi-supervised teacher model | (B, 1, H, W) |

## Training Process

**PHASE 1: Train Student + EMA**
- Input: Labeled depth (`depth_s`) + Unlabeled depth (`depth`)
- Student: Predict trên labeled
- Teacher: Predict trên unlabeled (with EMA)
- DPA: Depth-guided patch augmentation trên unlabeled
- Loss: Supervised + Consistency + DPA cutmix loss

**PHASE 2: Train Teacher Decoder**
- Input: Labeled depth
- Freeze: Teacher encoder
- Train: Teacher decoder only
- Reason: Có encoder được update tốt từ Phase 1

## Một Số Lưu Ý

1. **Depth Data:**
   - File `.npy` phải có cùng tên (không extension) với mask file
   - VD: `masks/image1.png` ↔ `depth_npy/image1.npy`

2. **Memory:**
   - Depth single-channel tiết kiệm ~3x memory so với RGB
   - Có thể tăng `batch_size` nếu muốn

3. **Hyperparameters:**
   - Các default giống như RGB training
   - Nếu cần, có thể điều chỉnh qua config file

4. **Evaluation:**
   - Metrics: Dice, IoU, Accuracy (trên unlabeled test set)
   - Cả student và teacher được evaluate

## Xử lý Lỗi

**Lỗi: "Depth file not found"**
- Kiểm tra folder `depth_npy` tồn tại
- Kiểm tra tên file khớp (không có extension)

**Lỗi: Shape mismatch**
- Depth array phải có cùng spatial resolution với mask
- Transform sẽ resize cả masks và depth

**Lỗi: Out of Memory**
- Giảm `batch_size` trong config
- Giảm `total_iter` hoặc số epochs

## So Sánh với RGB-Depth Training

| Yếu tố | RGB-Depth | Depth-Only |
|--------|-----------|-----------|
| Input channels | 4 (3 RGB + 1 D) | 1 |
| Memory | ~300MB/batch | ~100MB/batch |
| Model | `DepthFusion_*` | `DepthOnly_*` |
| DPA module | Dùng depth map | Dùng depth input |
| Training time | Lâu hơn | Nhanh hơn |

## Contact & Debug

Nếu gặp vấn đề:
1. Kiểm tra folder structure
2. Verify file `.npy` valid: `np.load(path).shape`
3. Xem log file: `logs/depthOnly.json`
