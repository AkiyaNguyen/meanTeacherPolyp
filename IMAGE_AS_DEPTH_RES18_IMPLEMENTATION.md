# Image-as-Depth + ResNet18 Depth Encoder

## Muc tieu

Che do nay huan luyen Mean Teacher ma khong can load depth map tu disk:

- Nhanh depth trong teacher dung cung input RGB image.
- Depth encoder cua teacher duoc thay bang ResNet18.
- Nap pretrained depth encoder tu `pretrained/encoder.pth`.

## Files chinh

- `emaEncoderOnlyTrain_imageAsDepth_res18.py`
- `utils/build_dataset_image_as_depth.py`
- `cfg/emaEncoderOnly_imageAsDepth_res18.yaml`

## Cach hoat dong

1. Dataset builder moi chi load `image` + `mask`, khong load `depth`.
2. Student van la model segmentation RGB (`ResNet34U_f`).
3. Teacher la model fusion (`DepthFusion_ResNet34U_f_EMAEncoderOnly`), nhung goi theo kieu:
   - `tea_model(img, img)`
4. Truoc khi train, teacher `depth_encoder` duoc thay thanh `encoder18` va load weight tu checkpoint.
5. EMA update giu nguyen tren `teacher.rgb_encoder` tu student encoder.

## Data flow

### Train phase 1

- Labeled batch: supervised loss tren student output.
- Unlabeled batch:
  - Teacher predict voi shared input image cho 2 nhanh.
  - DPA cutmix duoc tinh tren image tensor (thay vi depth tensor that).
  - Student consistency loss + cutmix loss.

### Train phase 2

- Freeze `teacher.rgb_encoder`.
- Train phan con lai cua teacher voi labeled batch.
- Van dung `tea_model(labeled_img, labeled_img)`.

## Config can dung

Dung file:

- `cfg/emaEncoderOnly_imageAsDepth_res18.yaml`

Thong so quan trong:

- `data.require_depth: false`
- `data.depth_dirname: ''`
- `data.test.depth_dirname: ''`
- `Trainer.depth_encoder_ckpt: pretrained/encoder.pth`

## Cach chay

```bash
python emaEncoderOnlyTrain_imageAsDepth_res18.py --config cfg/emaEncoderOnly_imageAsDepth_res18.yaml --optuna_trial_times 0
```

Override checkpoint neu can:

```bash
python emaEncoderOnlyTrain_imageAsDepth_res18.py --config cfg/emaEncoderOnly_imageAsDepth_res18.yaml Trainer.depth_encoder_ckpt=pretrained/encoder.pth
```

## Troubleshooting

1. Loi khong tim thay checkpoint:
   - Kiem tra file `pretrained/encoder.pth` ton tai.

2. Loi map state_dict:
   - Checkpoint co the dung key prefix khac.
   - Script da thu cac prefix: `''`, `encoder1.`, `depth_encoder.`, `encoder.`.

3. Dataloader rong:
   - Kiem tra `data.root`, `data.data2_dir`, `images/`, `masks/`.

4. Muon quay lai che do co depth that:
   - Dung script + config goc (`emaEncoderOnlyTrain.py`, `cfg/emaEncoderOnly.yaml`).
