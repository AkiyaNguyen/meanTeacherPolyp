# THAY DOI VE DEPTH-ONLY TRAINING

Cap nhat: 2026-03-28

## Tong quan

Da bo sung pipeline train depth-only day du, dung du lieu depth dang `.npy` va mask, khong can RGB input.

## Files da cap nhat

### 1) `models/ResUNet.py` [MODIFIED]

Them 3 class cho depth-only:

- `DepthOnlyEncoder` (line 649)
  - Input: `(B, 1, H, W)`
  - Tu dong lap channel de phu hop backbone ResNet34
  - Output: 5 muc feature encoder `(e1..e5)`

- `DepthOnlyResNet34U_f` (line 686)
  - Model student depth-only
  - Output segmentation 1 channel

- `DepthOnlyResNet34U_f_EMAEncoderOnly` (line 723)
  - Model teacher depth-only
  - Dung trong Mean Teacher, update EMA tren encoder

### 2) `data/dataset.py` [MODIFIED]

Them `DepthOnlyDataset` (line 114).

Tinh nang chinh:

- Load depth tu `.npy`
- Chuan hoa depth ve `[0, 255]`
- Train augment: flip, rotate, zoom, color jitter, blur
- Return keys: `depth_s`, `depth`, `label`, `id`

### 3) `utils/build_dataset_depth_only.py` [CREATED]

Noi dung chinh:

- `build_dataset_depth_only(cfg)` (line 11)
- `DepthOnlyImageFolderDataset` (line 126)
- Ho tro chia train/val/test
- Ho tro `TwoStreamBatchSampler` cho labeled/unlabeled

### 4) `depthOnlyTrain.py` [CREATED]

Script train chinh depth-only:

- `DepthOnly_MT_Trainer_EMAEncoderOnly` (line 25)
- `DepthOnlyEvalHook_EMAEncoderOnly` (line 169)
- CLI co `--config` va `--optuna_trial_times` (line 298+)

Flow train:

- Phase 1: train student + consistency + DPA + update EMA encoder
- Phase 2: freeze teacher encoder, train teacher tren labeled batch

### 5) `cfg/depthOnly.yaml` [CREATED]

Config mau cho depth-only. Cac key quan trong:

```yaml
data:
  dataset: DepthOnlyDataset
  depth_dirname: depth_npy
  test:
    depth_dirname: depth_npy
```

### 6) `data/__init__.py` [MODIFIED]

Da export:

```python
from .dataset import kvasir_SEG, DepthOnlyDataset
```

### 7) `DEPTH_ONLY_TRAINING_GUIDE.md` [CREATED]

Tai lieu huong dan chuan bi du lieu chi tiet.

### 8) `test_depth_only_setup.py` [CREATED]

Script check nhanh import, khoi tao model, forward pass va config ton tai.

## Cau truc du lieu mong doi

```text
<data_root>/
  TrainDataset/
    masks/
      image1.png
      image2.png
    depth_npy/
      image1.npy
      image2.npy
  TestDataset/
    Kvasir/
      masks/
      depth_npy/
```

Luu y:

- Ten file `.npy` phai trung stem voi mask (`image1.png` <-> `image1.npy`)
- Shape chap nhan: `(H, W)` hoac `(H, W, 1)`
- Neu depth la hang so (max == min), dataset se tao map 0 de tranh loi normalize

## Cach chay

### 1) Kiem tra setup

```bash
python test_depth_only_setup.py
```

### 2) Chay train co ban

```bash
python depthOnlyTrain.py --config cfg/depthOnly.yaml
```

### 3) Override nhanh bang CLI

```bash
python depthOnlyTrain.py --config cfg/depthOnly.yaml data.root=../polypdepth_dataset/ data.depth_dirname=depth_npy data.test.depth_dirname=depth_npy
```

### 4) Chay voi Optuna

```bash
python depthOnlyTrain.py --config cfg/depthOnly.yaml --optuna_trial_times 4
```

## Logging va output

- Log train: `logs/depthOnly.json`
- Metrics chinh:
  - `phase1_labeled_loss`
  - `phase1_unlabeled_depth_loss`
  - `phase1_unlabeled_depth_cutmix_loss`
  - `phase1_consistency_weight`
  - `phase2_teacher_labeled_loss`
  - `val_stu_Dice`, `val_tea_depth_Dice`
  - `test_stu_Dice`, `test_tea_depth_Dice`
  - `test_stu_IoU`, `test_tea_depth_IoU`
- Checkpoint luu theo hook config (`ckpt/` hoac thu muc duoc cau hinh)

## Checklist truoc khi train

- [ ] Da tao du thu muc `depth_npy`
- [ ] Tat ca `.npy` co ten khop mask
- [ ] Da set dung `data.root`
- [ ] Da set dung `data.depth_dirname` va `data.test.depth_dirname`
- [ ] Chay pass `python test_depth_only_setup.py`

## Troubleshooting nhanh

1. Loi khong tim thay file depth:
   - Kiem tra duong dan va ten file `.npy`
   - Dam bao `depth_dirname` khop ten folder that

2. Loi dataloader rong:
   - Kiem tra trong `masks/` co file `.png/.jpg/.jpeg`
   - Kiem tra `data.root` + `data.data2_dir`

3. Kiem tra nhanh du lieu depth:

```bash
python -c "import numpy as np; x=np.load('TrainDataset/depth_npy/image1.npy'); print(x.shape, x.dtype, x.min(), x.max())"
```

4. Kiem tra key config depth:

```bash
python -c "import yaml; c=yaml.safe_load(open('cfg/depthOnly.yaml','r',encoding='utf-8')); print(c['data']['depth_dirname'], c['data']['test']['depth_dirname'])"
```

## Tai lieu lien quan

- `DEPTH_ONLY_TRAINING_GUIDE.md`
- `test_depth_only_setup.py`
- `cfg/depthOnly.yaml`

## Next steps

1. Chuan hoa ten file mask/depth cho dong bo 1-1.
2. Chay script test setup.
3. Chay train depth-only voi config da override duong dan thuc te.
4. Theo doi `logs/depthOnly.json` va metric Dice/IoU theo tung epoch.
