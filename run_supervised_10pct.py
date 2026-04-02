"""
Fully supervised training for ResNet34U_f with only 10% labeled images.
Same data split as Mean Teacher (first N% as labeled); no teacher, no consistency loss.
Style aligned with emaEncoderOnlyTrain.py (training(cfg, trial), Optuna, score_criteria).
"""

from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook, HookBase
import typing
import argparse
import copy
from utils.hook import ExtendMLFlowLoggerHook
from utils import loss
from utils.common import *
import torch
import models
from torch.optim.lr_scheduler import LambdaLR
import optuna
from utils.build_dataset_supervised import build_dataset_supervised
from test.eval import evaluate


class SupervisedTrainer(Trainer):
    """Single-model fully supervised trainer (no teacher, no consistency)."""

    def __init__(self, model, train_dataloader, optimizer, scheduler, num_epochs, class_criterion, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_criterion = class_criterion

    def _start_train_mode(self) -> None:
        self.model.train()

    def run_step_(self) -> None:
        self.model.train()
        device = next(self.model.parameters()).device
        step_info = {'loss': []}

        for batch_id, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            img_s = data['image_s'].to(device)
            label = data['label'].to(device)

            pred = self.model(img_s)
            loss = self.class_criterion(pred, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            step_info['loss'].append(loss.item())
            self.scheduler.step()

        self._add_info({k: np.mean(v) for k, v in step_info.items()})

    def get_Trainer_ckpt(self) -> dict:
        """Checkpoint for SmartSaveHook: single model state."""
        return {'model': self.model.state_dict()}


class SupervisedEvalHook(EvalHook):
    """Eval hook for a single model (no teacher)."""

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
                output = self.trainer.model(img)
                cur = evaluate(output, gt)
                for key, value in cur.items():
                    metrics[key].append(value)

        return {self.prefix + key: np.mean(value) for key, value in metrics.items()}

    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.eval_every_epoch == 0:
            result = self._run_validation()
            self.trainer._add_info(result)


class StopTrainAtEpoch(HookBase):
    def __init__(self, trainer: Trainer, stop_at_epoch: int) -> None:
        super().__init__(trainer)
        self.stop_at_epoch = stop_at_epoch

    def after_train_epoch(self) -> None:
        if self.trainer.current_epoch + 1 >= self.stop_at_epoch:
            self.trainer.stop_training()
            print(f"Training stopped at epoch {self.stop_at_epoch}")


class SmartSaveHook(HookBase):
    """Save best checkpoint by criteria (single model); periodic and final save."""

    def __init__(self, trainer: Trainer, save_dir: str, max_save_epoch_interval: int, save_name: str, criteria: str) -> None:
        super().__init__(trainer)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.max_save_epoch_interval = max_save_epoch_interval
        self.save_name = save_name
        self.criteria = criteria
        self.patience = 0
        self.best_record = None
        self.has_improved = False
        self.ckpt = None

    def after_train_epoch(self) -> None:
        latest = self.trainer.info_storage.latest_info()
        self.patience += 1
        if self.criteria not in latest:
            return
        criteria_value = latest[self.criteria]
        if self.best_record is None or criteria_value > self.best_record:
            self.best_record = criteria_value
            self.has_improved = True
            self.ckpt = copy.deepcopy(self.trainer.get_Trainer_ckpt())
        if self.patience >= self.max_save_epoch_interval and self.has_improved:
            path = os.path.join(self.save_dir, f"{self.save_name}_epoch{self.trainer.current_epoch + 1}.pth")
            torch.save(self.ckpt, path)
            print(f"Model saved at {path}")
            self.has_improved = False
            self.patience = 0

    def after_train(self) -> None:
        if self.ckpt is not None:
            path = os.path.join(self.save_dir, f"final_{self.save_name}.pth")
            torch.save(self.ckpt, path)
            print(f"Final model saved at {path}")


def training(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
    if trial is not None:
        sweep_config = cfg.get('hyperparameter_sweeping', {})
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
    nEpoch = cfg.get('total_iter') // iters_per_epoch
    total_iter = cfg.get('total_iter')
    print(f"Total iterations: {total_iter} | Iters/epoch: {iters_per_epoch} => nEpoch: {nEpoch}")

    model_name = cfg.get('model.name', 'ResNet34U_f')
    num_classes = cfg.get('model.num_channels_output', 1)
    model = getattr(models, model_name)(num_classes=num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get('optimizer.lr'),
        momentum=cfg.get('optimizer.momentum'),
        weight_decay=cfg.get('optimizer.weight_decay')
    )
    scheduler_power = float(cfg.get('scheduler.power', 0.9))
    scheduler = LambdaLR(
        optimizer,
        lambda e: max(0.0, 1.0 - pow(min(e, total_iter) / total_iter, scheduler_power))
    )

    class_criterion = getattr(loss, cfg.get('Trainer.class_criterion', 'BCEDiceLoss'))()

    trainer = SupervisedTrainer(model, train_dataloader, optimizer, scheduler, nEpoch, class_criterion)

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(SupervisedEvalHook, eval_data_loader=val_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)), prefix='val_')
    hook_builder(SupervisedEvalHook, eval_data_loader=test_dataloader,
                 eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)), prefix='test_')
    # hook_builder(SmartSaveHook,
    #              save_dir=cfg.get('Hook.SmartSaveHook.save_dir', 'save_dir/'),
    #              max_save_epoch_interval=int(cfg.get('Hook.SmartSaveHook.max_save_epoch_interval', 50)),
    #              save_name=cfg.get('Hook.SmartSaveHook.save_name', 'supervised_10pct'),
    #              criteria=cfg.get('Hook.SmartSaveHook.criteria', 'test_Dice'))
    hook_builder(LoggerHook, logger_file=cfg.get('Hook.LoggerHook.logger_file', 'logs/supervised.json'))
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
    if cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch', None) is not None:
        hook_builder(StopTrainAtEpoch, stop_at_epoch=int(cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch')))

    trainer.train()

    criteria = cfg.get('score_criteria', 'test_Dice')
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f"Criteria {criteria} does not exist in info_storage")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fully supervised ResNet34U_f with 10% labeled (same style as emaEncoderOnlyTrain).')
    parser.add_argument('--optuna_trial_times', type=int, default=0, help='Optuna trials; 0 = no Optuna.')
    parser.add_argument('--config', type=str, default='cfg/supervised_10pct.yaml', help='Path to YAML config')
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

    print("Training completed!")

# !cd /kaggle/working/meanTeacherPolyp && \
#     python run_supervised_10pct.py \
#                     --optuna_trial_times 4\
#                     data.root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG data.data2_dir='Train' \
#                     data.test.dataset_root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Test \
#                     data.dataset=kvasir_SEG \
#                     Hook.ExtendMLFlowLoggerHook.run_name='supervised_10pct' \
#                     Hook.StopTrainAtEpoch.stop_at_epoch=200 \
#                     Hook.ExtendMLFlowLoggerHook.experiment_name='supervised_10pct' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.kaggle_run_link='https://www.kaggle.com/code/akiyanguyen/kagglerunningtemplate/edit' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.version=3

# !cd /kaggle/working/meanTeacherPolyp && \
#     python run_supervised_10pct.py \
#                     --optuna_trial_times 3\
#                     data.root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG data.data2_dir='Train' \
#                     data.test.dataset_root=/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Test \
#                     data.dataset=kvasir_SEG \
#                     Hook.ExtendMLFlowLoggerHook.run_name='supervised_100pct' \
#                     Hook.StopTrainAtEpoch.stop_at_epoch=300 \
#                     Hook.ExtendMLFlowLoggerHook.max_save_epoch_interval=100 \
#                     Hook.data.val_split_perc=0 \ 
#                     Hook.data.labeled_perc=100 \
#                     Hook.ExtendMLFlowLoggerHook.experiment_name='supervised_100pct' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.kaggle_run_link='https://www.kaggle.com/code/akiyanguyen/kagglerunningtemplate/edit' \
#                     Hook.ExtendMLFlowLoggerHook.meta_info.version=3