# VectorLoRA Run Implementation Details

## 1. Goal
This run adapts Vector-LoRA to the DAv2 backbone used inside the teacher model `DAv2Fusion_ResNet34U_f_EMAEncoderOnly`, while keeping the original Mean Teacher phase logic intact.

## 2. Main Files
- Training script: DEMT_DAv2_VectorLoRA.py
- Config: cfg/DEMT_DAv2_VectorLoRA.yaml
- LoRA utilities: utils/lora_utils.py
- Teacher architecture: models/ResUNet.py (class DAv2Fusion_ResNet34U_f_EMAEncoderOnly)

## 3. Model Architecture Context
- Student: ResNet34U_f (RGB only)
- Teacher: DAv2Fusion_ResNet34U_f_EMAEncoderOnly
  - RGB encoder branch
  - DAv2 backbone branch (from Hugging Face Depth Anything V2 backbone)
  - Projection + fusion + decoder

The teacher uses DAv2 features extracted from RGB input; no external depth map is passed to this teacher path.

## 4. Vector-LoRA Design Used in This Run
### 4.1 Fused-QKV path
`VectorLoRA_QKV` wraps one fused `nn.Linear` qkv layer:
- keeps original frozen qkv
- adds LoRA to Q and V only
- K remains unchanged
- scaling = alpha / rank

### 4.2 Split-QV fallback path
Some DAv2 variants expose attention as separate projections (`query`, `key`, `value`) rather than one `qkv`.

`VectorLoRA_Linear` is used as fallback wrapper:
- injects into `query` and `value`
- does not inject into `key`

### 4.3 Rank schedule (Vector-LoRA style)
- blocks 0-2: rank 16
- blocks 3-5: rank 8
- blocks 6-8: rank 4
- blocks 9-11: rank 2
- deeper blocks fallback to rank 2

## 5. Injection Logic
Function: `inject_vector_lora_into_dav2(model, alpha=16.0)`

Discovery strategy:
1. Try explicit block paths:
   - blocks
   - pretrained.blocks
   - backbone.blocks
   - backbone.pretrained.blocks
   - model.blocks
   - model.pretrained.blocks
   - module.blocks
   - module.pretrained.blocks
2. If not found, scan all modules for:
   - `attn.qkv` (fused case)
   - `*.query` and `*.value` (split case)

After wrapping:
- freeze all parameters
- re-enable only LoRA params (`lora_*`)
- print injected count and trainable ratio summary

## 6. Important Runtime Fix Applied
### Device mismatch fix
A runtime error occurred because LoRA params were created on CPU after the model had already been moved to CUDA.

Fix applied in `utils/lora_utils.py`:
- LoRA params are now initialized with the same device and dtype as the wrapped base layer weight (`qkv.weight` or `linear.weight`).

This prevents:
- `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`

## 7. Training Script Integration
In `DEMT_DAv2_VectorLoRA.py`:
1. import `inject_vector_lora_into_dav2`
2. build student and teacher
3. freeze teacher `dav2_encoder` base params
4. if `lora_params.enable_lora` is true, inject LoRA into `tea_model.dav2_encoder`
5. build teacher optimizer from dynamic trainable params (`p.requires_grad == True`)
6. keep baseline phase-1 and phase-2 loop/loss structure

## 8. Config for This Run
`cfg/DEMT_DAv2_VectorLoRA.yaml` adds:
- lora_params.enable_lora: True
- lora_params.base_rank: 16
- lora_params.alpha: 16

Other baseline hyperparameters are kept unchanged.

## 9. Run Command Example
Use the same command pattern:

```bash
python .\DEMT_DAv2_VectorLoRA.py \
  --optuna_trial_times 0 \
  data.root=archive/polypDataset_final1/kvasir_SEG \
  data.data2_dir='Train' \
  data.test.dataset_root=archive/polypDataset_final1/kvasir_SEG/Test \
  data.dataset=kvasir_SEG \
  Hook.ExtendMLFlowLoggerHook.run_name='DEMT_DAv2Fusion_addDepthTrainSignal_ton3' \
  nEpoch=300 \
  Hook.ExtendMLFlowLoggerHook.experiment_name='DEMT_DAv2Fusion_addDepthTrainSignal' \
  seed=1111
```

## 10. Quick Verification Checklist
- Check console for Vector-LoRA injection summary lines.
- Confirm non-zero trainable params after injection.
- Confirm training starts past first teacher forward without device mismatch.
- Confirm loss values are logged for phase1 and phase2.
- Confirm checkpoints are saved by logger hook.

## 11. Notes
- If a new DAv2 release changes attention module names, update fallback scanner patterns in `utils/lora_utils.py`.
- If memory is tight, reduce rank schedule or disable LoRA by config.
