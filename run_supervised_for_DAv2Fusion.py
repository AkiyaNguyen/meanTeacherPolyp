from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook
import copy
from utils.hook import ExtendMLFlowLoggerHook

from test.eval import evaluate
import typing
import argparse

from utils.common import *
from utils import loss
import torch
import optuna
import models

from torch.optim.lr_scheduler import LambdaLR
from utils.build_dataset_supervised import build_dataset_supervised


def _pred_for_loss_and_metrics(raw):
    """Normalize model outputs to a single segmentation tensor."""
    if isinstance(raw, dict):
        raw = raw.get('rgb_depth')
        if raw is None:
            raise ValueError("Model returned a dict without 'rgb_depth'.")
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return raw


class SupervisedTrainer_DAv2Fusion(Trainer):
    """Fully supervised trainer for DAv2Fusion_ResNet34U_f_EMAEncoderOnly."""

    def __init__(self, model, train_dataloader, optimizer, scheduler, num_epochs, class_criterion, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_criterion = class_criterion

    def _start_train_mode(self) -> None:
        self.model.train()

    def get_Trainer_ckpt(self) -> dict:
        return {
            'model': self.model.state_dict(),
            'current_epoch': self.current_epoch + 1,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_Trainer_ckpt(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.current_epoch = state_dict['current_epoch']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def run_step_(self) -> None:
        self.model.train()
        device = next(self.model.parameters()).device
        step_info = {'loss': []}

        for _, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            img_s, label = data['image_s'], data['label']
            img_s, label = img_s.to(device), label.to(device)

            pred = _pred_for_loss_and_metrics(self.model(img_s))
            loss_value = self.class_criterion(pred, label)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            step_info['loss'].append(loss_value.item())

        self._add_info({k: np.mean(v) for k, v in step_info.items()})


class SupervisedEvalHook_DAv2Fusion(EvalHook):
    """Eval hook for single supervised DAv2Fusion model."""

    def __init__(self, trainer: Trainer, eval_data_loader: torch.utils.data.DataLoader, eval_every_epoch: int, prefix: str = '') -> None:
        super().__init__(trainer, eval_data_loader)
        self.eval_every_epoch = eval_every_epoch
        self.prefix = prefix
        assert self.eval_every_epoch >= 1, "eval_every_epoch must be at least 1"

    def _run_validation(self) -> dict[typing.Any, typing.Any]:
        assert hasattr(self.trainer, 'model'), "trainer must have model attribute"
        self.trainer.model.eval()
        device = next(self.trainer.model.parameters()).device
        metrics = {'ACC_overall': [], 'Dice': [], 'IoU': []}

        with torch.no_grad():
            for data in self.eval_data_loader:
                img = data['image'].to(device)
                gt = data['mask'].to(device)
                output = _pred_for_loss_and_metrics(self.trainer.model(img))
                cur_metrics = evaluate(output, gt)
                for key, value in cur_metrics.items():
                    metrics[key].append(value)

        return {self.prefix + key: np.mean(value) for key, value in metrics.items()}

    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.eval_every_epoch == 0:
            result = self._run_validation()
            self.trainer._add_info(result)


def training(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
    if trial is not None:
        sweep_config = cfg.get('hyperparameter_sweeping') or cfg.get('hyperparameters_sweeping', {})
        for key, settings in sweep_config.items():
            suggested_value = getattr(trial, settings['method'])(**settings['params'])
            cfg.set(key, suggested_value)

    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))

    train_dataloader, val_dataloader, test_dataloader = build_dataset_supervised(cfg)
    assert train_dataloader is not None, "train_dataloader is None"
    assert test_dataloader is not None, "test_dataloader is None"

    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    nEpoch = int(cfg.get('nEpoch', 300))
    total_iter = nEpoch * iters_per_epoch
    print(f"nEpoch: {nEpoch} | Iters/epoch: {iters_per_epoch} => total train steps: {total_iter}")

    model = getattr(
        models,
        cfg.get('model.name', 'DAv2Fusion_ResNet34U_f_EMAEncoderOnly')
    )(num_classes=cfg.get('model.num_channels_output', 1)).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get('optimizer.lr'),
        momentum=cfg.get('optimizer.momentum'),
        weight_decay=cfg.get('optimizer.weight_decay'),
    )
    scheduler_power = float(cfg.get('scheduler.power', 0.9))
    scheduler = LambdaLR(
        optimizer,
        lambda e: max(0.0, 1.0 - pow(min(e, total_iter) / total_iter, scheduler_power)),
    )

    class_criterion = getattr(loss, cfg.get('Trainer.class_criterion', 'WeightedBCEDiceLoss'))()
    trainer = SupervisedTrainer_DAv2Fusion(
        model, train_dataloader, optimizer, scheduler, nEpoch, class_criterion
    )
    if cfg.get('Trainer.load_ckpt_path', None) is not None:
        trainer.load_Trainer_ckpt(torch.load(cfg.get('Trainer.load_ckpt_path')))
        print(f"Loaded checkpoint from {cfg.get('Trainer.load_ckpt_path')}")
    else:
        print("No checkpoint loaded")

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(
            SupervisedEvalHook_DAv2Fusion,
            eval_data_loader=val_dataloader,
            eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)),
            prefix='val_',
        )
    hook_builder(
        SupervisedEvalHook_DAv2Fusion,
        eval_data_loader=test_dataloader,
        eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)),
        prefix='test_',
    )
    hook_builder(
        LoggerHook,
        logger_file=cfg.get('Hook.LoggerHook.logger_file', 'logs/dav2_fusion_train_supervised.json'),
    )
    hook_builder(
        ExtendMLFlowLoggerHook,
        local_dir_save_ckpt=cfg.get('Hook.ExtendMLFlowLoggerHook.local_dir_save_ckpt'),
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
    # if cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch', None) is not None:
    #     hook_builder(StopTrainAtEpoch, stop_at_epoch=int(cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch')))

    trainer.train()

    criteria = cfg.get('score_criteria', 'test_Dice')
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f"Criteria {criteria} does not exist in info_storage")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fully supervised DAv2Fusion training.')
    parser.add_argument('--optuna_trial_times', type=int, default=0, help='Optuna trials; 0 = no Optuna.')
    parser.add_argument('--config', type=str, default='cfg/dav2_fusion_train_supervised.yaml', help='Path to YAML config')
    args, unknown = parser.parse_known_args()
    cfg = Config(config_file=args.config, cli_overrides=unknown)

    if args.optuna_trial_times == 0:
        score = training(cfg)
        print(f"Score: {score}")
    else:
        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)
            return training(trial_cfg, trial)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.optuna_trial_times)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


# !cd /kaggle/working/meanTeacherPolyp && \
#     python run_supervised_for_DAv2Fusion.py \
#                     --optuna_trial_times 4\
#                     data.root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG data.data2_dir='Train' \
#                     data.test.dataset_root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Test \
#                     data.dataset=kvasir_SEG \
#                     Hook.ExtendMLFlowLoggerHook.run_name='DAv2Fusion_train_supervised' \
#                     Hook.StopTrainAtEpoch.stop_at_epoch=200 \
#                     Hook.ExtendMLFlowLoggerHook.experiment_name='DAv2Fusion_train_supervised' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.kaggle_run_link='https://www.kaggle.com/code/akiyanguyen/kagglerunningtemplate/edit' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.version=3