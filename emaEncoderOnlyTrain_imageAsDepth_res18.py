from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook, StopTrainAtEpoch
from utils.hook import ExtendMLFlowLoggerHook
import copy
import os
from pathlib import Path

from test.eval import evaluate
import typing
import argparse

from utils.common import *
from utils.dpa import dpa
import torch
import torch.nn as nn
import optuna

from torch.optim.lr_scheduler import LambdaLR
from utils.ramps import sigmoid_rampup

from utils.build_dataset_image_as_depth import build_dataset_image_as_depth
from utils.loss import SoftmaxMSELoss, BCEDiceLoss


def resolve_checkpoint_path(ckpt_path: str) -> str:
    """
    Resolve checkpoint path robustly across Windows/Linux and common misconfigurations.

    Supports:
    - Absolute path
    - Relative to current working directory
    - Relative to repository/script directory
    - Mistyped leading slash like '/pretrained/encoder.pth'
    """
    raw = (ckpt_path or '').strip()
    if raw == '':
        raise ValueError('Empty checkpoint path for Trainer.depth_encoder_ckpt')

    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd().resolve()

    candidates = []
    p = Path(raw)

    if p.is_absolute():
        candidates.append(p)
        # Also try interpreting '/pretrained/...' as repo-relative path.
        stripped = raw.lstrip('/\\')
        if stripped:
            candidates.append(cwd_dir / stripped)
            candidates.append(script_dir / stripped)
    else:
        candidates.append(cwd_dir / p)
        candidates.append(script_dir / p)

    for cand in candidates:
        if cand.is_file():
            return str(cand)

    tried = ' | '.join(str(c) for c in candidates)
    raise FileNotFoundError(f'Encoder checkpoint not found: {ckpt_path}. Tried: {tried}')


def _unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ('state_dict', 'model', 'model_state_dict', 'net'):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        return checkpoint_obj
    raise ValueError('Unsupported checkpoint format for encoder weights')


def _sanitize_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        result[nk] = v
    return result


def _remap_key_candidates(key: str) -> list[str]:
    """Generate possible local key names for common checkpoint naming styles."""
    candidates = [key]

    # Strip outer wrappers frequently seen in checkpoints.
    prefixes = (
        'encoder.',
        'depth_encoder.',
        'backbone.',
        'model.',
        'resnet.',
    )
    stripped_variants = [key]
    for p in prefixes:
        if key.startswith(p):
            stripped_variants.append(key[len(p):])

    # Map canonical ResNet names -> local encoder18 names.
    mapping = {
        'conv1.': 'encoder1_conv.',
        'bn1.': 'encoder1_bn.',
        'layer1.': 'encoder2.',
        'layer2.': 'encoder3.',
        'layer3.': 'encoder4.',
        'layer4.': 'encoder5.',
    }

    for s in stripped_variants:
        candidates.append(s)
        for src, dst in mapping.items():
            if s.startswith(src):
                candidates.append(dst + s[len(src):])

    # De-duplicate while keeping order.
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _match_encoder_keys(
    raw_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    prefix_to_strip: str,
) -> dict[str, torch.Tensor]:
    matched = {}
    for k, v in raw_state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if prefix_to_strip and nk.startswith(prefix_to_strip):
            nk = nk[len(prefix_to_strip):]
        for cand in _remap_key_candidates(nk):
            if cand in target_state_dict and target_state_dict[cand].shape == v.shape:
                matched[cand] = v
                break
    return matched


def load_pretrained_resnet18_depth_encoder(tea_model: nn.Module, ckpt_path: str) -> None:
    # Replace teacher depth branch with ResNet18 encoder.
    tea_model.depth_encoder = models.encoder18(num_classes=None)  # type: ignore[attr-defined]

    resolved_ckpt_path = resolve_checkpoint_path(ckpt_path)
    checkpoint = torch.load(resolved_ckpt_path, map_location='cpu')
    raw_sd = _sanitize_keys(_unwrap_state_dict(checkpoint))
    target_sd = tea_model.depth_encoder.state_dict()  # type: ignore[attr-defined]

    candidate_prefixes = ['', 'encoder1.', 'depth_encoder.', 'encoder.', 'model.', 'backbone.', 'resnet.']
    best_matched: dict[str, torch.Tensor] = {}
    best_prefix = ''
    for prefix in candidate_prefixes:
        matched = _match_encoder_keys(raw_sd, target_sd, prefix)
        if len(matched) > len(best_matched):
            best_matched = matched
            best_prefix = prefix

    if len(best_matched) == 0:
        raw_tensor_keys = [k for k, v in raw_sd.items() if isinstance(v, torch.Tensor)]
        raw_sample = raw_tensor_keys[:15]
        target_sample = list(target_sd.keys())[:15]
        raise RuntimeError(
            'Could not map any checkpoint weights to depth_encoder. '
            f'Checkpoint: {ckpt_path} | '
            f'raw_tensor_keys_sample={raw_sample} | '
            f'target_keys_sample={target_sample}'
        )

    missing, unexpected = tea_model.depth_encoder.load_state_dict(best_matched, strict=False)  # type: ignore[attr-defined]
    print(
        f'[DepthEncoder] Loaded ResNet18 weights from {resolved_ckpt_path} | '
        f'matched={len(best_matched)} | stripped_prefix="{best_prefix}" | '
        f'missing={len(missing)} unexpected={len(unexpected)}'
    )


class ImageAsDepth_MT_Trainer_EMAEncoderOnly(Trainer):
    """
    Mean Teacher trainer for image-as-depth mode:
    - No depth loading from dataset.
    - Teacher gets the same RGB image for both RGB and depth branches.
    - Teacher depth encoder is ResNet18 loaded from external checkpoint.
    """
    def __init__(self, stu_model, tea_model, train_dataloader, stu_optimizer, tea_optimizer, scheduler, num_epochs, ema_alpha,
                 consistency_rampup, consistency, fea_sim_weight: float, tea_scheduler=None, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.stu_model = stu_model
        self.tea_model = tea_model
        self.train_dataloader = train_dataloader
        self.stu_optimizer = stu_optimizer
        self.tea_optimizer = tea_optimizer
        self.scheduler = scheduler
        self.tea_scheduler = tea_scheduler
        self.ema_alpha = ema_alpha
        self.labeled_bs = self.train_dataloader.batch_sampler.primary_batch_size
        self.consistency_rampup = consistency_rampup
        self.consistency = consistency
        self.fea_sim_weight = fea_sim_weight
        self.class_criterion = BCEDiceLoss()
        self.consistency_criterion = SoftmaxMSELoss()
        self.dpa_loss = BCEDiceLoss()

    def _get_current_consistency_weight(self, global_step):
        return self.consistency * sigmoid_rampup(current=global_step, rampup_length=self.consistency_rampup)

    def _update_ema_variable(self, global_step, model_a: nn.Module, model_b: nn.Module):
        coeff = min(1 - 1 / (global_step + 1), self.ema_alpha)
        for tea_param, stu_param in zip(model_a.parameters(), model_b.parameters()):
            tea_param.data.mul_(coeff).add_(stu_param.data, alpha=1 - coeff)

    def get_Trainer_ckpt(self) -> dict:
        result = dict()
        result['stu_model'] = self.stu_model.state_dict()
        result['tea_model'] = self.tea_model.state_dict()
        result['current_epoch'] = self.current_epoch + 1
        result['stu_optimizer'] = self.stu_optimizer.state_dict()
        result['tea_optimizer'] = self.tea_optimizer.state_dict()
        result['scheduler'] = self.scheduler.state_dict()
        result['tea_scheduler'] = self.tea_scheduler.state_dict() if \
            hasattr(self, 'tea_scheduler') and self.tea_scheduler is not None else None
        return result

    def load_Trainer_ckpt(self, state_dict: dict) -> None:
        self.stu_model.load_state_dict(state_dict['stu_model'])
        self.tea_model.load_state_dict(state_dict['tea_model'])
        self.current_epoch = state_dict['current_epoch']
        self.stu_optimizer.load_state_dict(state_dict['stu_optimizer'])
        self.tea_optimizer.load_state_dict(state_dict['tea_optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        if state_dict.get('tea_scheduler') is not None and self.tea_scheduler is not None:
            self.tea_scheduler.load_state_dict(state_dict['tea_scheduler'])

    def _start_train_mode(self) -> None:
        self.stu_model.train()

    def run_step_(self) -> None:
        self.stu_model.train()
        self.tea_model.train()
        device = next(self.stu_model.parameters()).device

        phase1_info = {'labeled_loss': [], 'unlabeled_rgbd_loss': [],
                       'consistency_weight': [], 'unlabeled_rgbd_cutmix_loss': [], 'loss': []}
        phase2_info = {'teacher_labeled_loss': []}

        # Phase 1: train student + EMA on teacher rgb encoder only
        for batch_id, data in enumerate(self.train_dataloader):
            self.stu_optimizer.zero_grad()
            img_s, img, label = data['image_s'], data['image'], data['label']
            img_s, img, label = img_s.to(device), img.to(device), label.to(device)

            unlabeled_img = img[self.labeled_bs:]
            unlabeled_img_s = img_s[self.labeled_bs:]
            label = label[:self.labeled_bs]

            stu_pred = self.stu_model(img_s)
            labeled_stu = stu_pred[:self.labeled_bs]
            unlabeled_stu = stu_pred[self.labeled_bs:]

            with torch.no_grad():
                # Shared image input for both branches (no depth input).
                tea_output = self.tea_model(unlabeled_img, unlabeled_img)

            unlabeled_img_s_cutmix, ema_pred_u_cutmix = dpa(
                unlabeled_img, unlabeled_img_s, tea_output, beta=0.3, t=self.current_epoch, T=self.num_epochs
            )
            pred_u_cutmix = self.stu_model(unlabeled_img_s_cutmix)
            loss_consist_rgbd_cutmix = self.dpa_loss(pred_u_cutmix, ema_pred_u_cutmix)

            loss_sup = self.class_criterion(labeled_stu, label)
            loss_consist_rgbd = self.consistency_criterion(unlabeled_stu, tea_output)
            consistency_weight = self._get_current_consistency_weight(
                global_step=batch_id + self.current_epoch * len(self.train_dataloader)
            )
            total_loss = loss_sup + consistency_weight * loss_consist_rgbd + loss_consist_rgbd_cutmix

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), max_norm=1.0)
            self.stu_optimizer.step()
            self._update_ema_variable(
                global_step=batch_id + self.current_epoch * len(self.train_dataloader),
                model_a=self.tea_model.rgb_encoder,
                model_b=self.stu_model.encoder1,
            )

            phase1_info['labeled_loss'].append(loss_sup.item())
            phase1_info['unlabeled_rgbd_loss'].append(loss_consist_rgbd.item())
            phase1_info['unlabeled_rgbd_cutmix_loss'].append(loss_consist_rgbd_cutmix.item())
            phase1_info['consistency_weight'].append(consistency_weight)
            phase1_info['loss'].append(total_loss.item())
            self.scheduler.step()

        self._add_info({f'phase1_{k}': np.mean(v) for k, v in phase1_info.items()})

        # Phase 2: train teacher branch/fusion/decoder, freeze teacher rgb encoder
        for _, data in enumerate(self.train_dataloader):
            self.tea_optimizer.zero_grad()
            img, label = data['image'], data['label']
            img, label = img.to(device), label.to(device)

            labeled_img = img[:self.labeled_bs]
            label = label[:self.labeled_bs]

            for param in self.tea_model.rgb_encoder.parameters():
                param.requires_grad_(False)

            tea_labeled_rgbd_output = self.tea_model(labeled_img, labeled_img)
            loss_tea_sup = self.class_criterion(tea_labeled_rgbd_output, label)
            loss_tea_sup.backward()

            tea_param = self.tea_optimizer.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(tea_param, max_norm=1.0)
            self.tea_optimizer.step()

            phase2_info['teacher_labeled_loss'].append(loss_tea_sup.item())
            for param in self.tea_model.parameters():
                param.requires_grad_(True)

        if self.tea_scheduler is not None:
            self.tea_scheduler.step()
        self._add_info({f'phase2_{k}': np.mean(v) for k, v in phase2_info.items()})


class MeanTeacherEvalHook_ImageAsDepth(EvalHook):
    """Eval hook for image-as-depth mode (no depth key required from dataloader)."""
    def __init__(self, trainer: Trainer, eval_data_loader: torch.utils.data.DataLoader, eval_every_epoch: int, prefix: str = '') -> None:
        super().__init__(trainer, eval_data_loader)
        self.eval_every_epoch = eval_every_epoch
        self.prefix = prefix
        assert self.eval_every_epoch >= 1, 'eval_every_epoch must be at least 1'

    def _run_validation(self) -> dict[typing.Any, typing.Any]:
        assert hasattr(self.trainer, 'stu_model') and hasattr(self.trainer, 'tea_model'), \
            'trainer must have stu_model and tea_model attributes'
        self.trainer.stu_model.eval()
        self.trainer.tea_model.eval()
        device = next(self.trainer.stu_model.parameters()).device
        metrics = {
            'stu_ACC_overall': [],
            'tea_rgbd_ACC_overall': [],
            'stu_Dice': [],
            'tea_rgbd_Dice': [],
            'stu_IoU': [],
            'tea_rgbd_IoU': [],
        }
        with torch.no_grad():
            for data in self.eval_data_loader:
                img = data['image'].to(device)
                gt = data['mask'].to(device)

                stu_output = self.trainer.stu_model(img)
                cur_stu_metrics = evaluate(stu_output, gt)

                tea_output = self.trainer.tea_model(img, img)
                cur_tea_rgbd_metrics = evaluate(tea_output, gt)

                for key, value in cur_stu_metrics.items():
                    metrics['stu_' + key].append(value)
                for key, value in cur_tea_rgbd_metrics.items():
                    metrics['tea_rgbd_' + key].append(value)

        return {self.prefix + key: np.mean(value) for key, value in metrics.items()}

    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.eval_every_epoch == 0:
            result = self._run_validation()
            self.trainer._add_info(result)


def training(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
    if trial is not None:
        sweep_config = cfg.get('hyperparameter_sweeping', {})
        for key, settings in sweep_config.items():
            suggested_value = getattr(trial, settings['method'])(**settings['params'])
            cfg.set(key, suggested_value)

    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))

    train_dataloader, val_dataloader, test_dataloader = build_dataset_image_as_depth(cfg)
    assert train_dataloader is not None, 'train_dataloader is None'
    assert test_dataloader is not None, 'test_dataloader is None'

    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError('Dataloader is empty!')
    nEpoch = cfg.get('total_iter') // iters_per_epoch
    total_iter = cfg.get('total_iter')
    print(f'Total iterations: {total_iter} | Iters/epoch: {iters_per_epoch} => nEpoch: {nEpoch}')

    stu_model = getattr(models, cfg.get('model.stu_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)
    tea_model = getattr(models, cfg.get('model.tea_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)

    trainer_cfg = cfg.get('Trainer', {})
    depth_encoder_ckpt = trainer_cfg.get('depth_encoder_ckpt', 'pretrained/encoder.pth')
    # depth_encoder_ckpt = cfg.get('Trainer.depth_encoder_ckpt', 'pretrained/encoder.pth')
    load_pretrained_resnet18_depth_encoder(tea_model, depth_encoder_ckpt)
    tea_model = tea_model.to(device)

    optimizer = torch.optim.SGD(stu_model.parameters(), lr=cfg.get('optimizer.lr'),
                                momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    tea_optimizer = torch.optim.SGD(tea_model.parameters(), lr=cfg.get('optimizer.lr'),
                                    momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    scheduler_power = float(cfg.get('scheduler.power'))
    scheduler = LambdaLR(optimizer, lambda e: max(0.0, 1.0 - pow(min(e, total_iter) / total_iter, scheduler_power)))
    tea_scheduler = LambdaLR(tea_optimizer, lambda e: max(0.0, 1.0 - pow(min(e, nEpoch) / nEpoch, scheduler_power)))

    trainer = ImageAsDepth_MT_Trainer_EMAEncoderOnly(
        stu_model, tea_model, train_dataloader,
        optimizer, tea_optimizer,
        scheduler, nEpoch,
        ema_alpha=float(cfg.get('Trainer.ema_decay', 0.999)),
        consistency_rampup=float(cfg.get('Trainer.consistency_rampup')),
        consistency=float(cfg.get('Trainer.consistency')),
        fea_sim_weight=float(cfg.get('Trainer.fea_sim_weight', 0.5)),
        tea_scheduler=tea_scheduler,
    )

    if cfg.get('Trainer.load_ckpt_path', None) is not None:
        trainer.load_Trainer_ckpt(torch.load(cfg.get('Trainer.load_ckpt_path')))
        print(f"Loaded checkpoint from {cfg.get('Trainer.load_ckpt_path')}")
    else:
        print('No training checkpoint loaded')

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(MeanTeacherEvalHook_ImageAsDepth, eval_data_loader=val_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='val_')
    hook_builder(MeanTeacherEvalHook_ImageAsDepth, eval_data_loader=test_dataloader,
                 eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='test_')

    hook_builder(ExtendMLFlowLoggerHook, local_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.local_dir_save_ckpt'),
                 dagshub_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_dir_save_ckpt'),
                 max_save_epoch_interval=int(cfg.get('Hook.ExtendMLFlowLoggerHook.max_save_epoch_interval')),
                 criteria=cfg.get('Hook.ExtendMLFlowLoggerHook.criteria'),
                 dagshub_destination_src_file=str(cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_destination_src_file')),
                 list_src_dir_files=list(cfg.get('Hook.ExtendMLFlowLoggerHook.list_src_dir_files')),
                 dagshub_meta_dir=str(cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_meta_dir')),
                 meta_info=dict(cfg.get('Hook.ExtendMLFlowLoggerHook.meta_info')),
                 dagshub_repo_owner=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_repo_owner'),
                 dagshub_repo_name=cfg.get('Hook.ExtendMLFlowLoggerHook.dagshub_repo_name'),
                 experiment_name=str(cfg.get('Hook.ExtendMLFlowLoggerHook.experiment_name')),
                 dir_save_plot=cfg.get('Hook.ExtendMLFlowLoggerHook.dir_save_plot'),
                 logging_fields=list(cfg.get('Hook.ExtendMLFlowLoggerHook.logging_fields')),
                 run_name=cfg.get('Hook.ExtendMLFlowLoggerHook.run_name'),
                 cfg=cfg,
                 )
    hook_builder(LoggerHook, logger_file='logs/imageAsDepth_res18_encoder.json')
    hook_builder(StopTrainAtEpoch, stop_at_epoch=int(cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch')))

    trainer.train()

    criteria = cfg.get('score_criteria')
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f'Criteria {criteria} does not exist in info_storage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train with shared image input for RGB/depth branches and ResNet18 pretrained depth encoder.'
    )
    parser.add_argument('--optuna_trial_times', type=int, default=0, help='Optuna trials; 0 = no Optuna.')
    parser.add_argument('--config', type=str, default='cfg/emaEncoderOnly_imageAsDepth_res18.yaml', help='Path to YAML config')
    args, unknown = parser.parse_known_args()
    cfg = Config(config_file=args.config, cli_overrides=unknown)

    if args.optuna_trial_times == 0:
        score = training(cfg)
        print(f'Score: {score}')
    else:
        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)
            return training(trial_cfg, trial)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.optuna_trial_times)
        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params:')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
