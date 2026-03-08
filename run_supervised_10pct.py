"""
Fully supervised training for ResNet34U_f with only 10% labeled images.
Same data split as Mean Teacher (first 10% as labeled); no teacher, no consistency loss.
"""

from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook, HookBase, MLFlowLoggerHook
from test.eval import evaluate, ImageFolderDataset
from data.transform import Resize, ToTensor
from torchvision import transforms
import typing
import argparse
from torch.utils.data import Subset

from utils.common import *
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class SupervisedTrainer(Trainer):
    """Single-model fully supervised trainer (no teacher, no consistency)."""

    def __init__(self, model, train_dataloader, optimizer, scheduler, num_epochs, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_criterion = nn.BCELoss()

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


class FrequentSaveModelSingle(HookBase):
    """Save a single model checkpoint every N epochs."""

    def __init__(self, trainer: Trainer, save_dir: str, save_every_epoch: int, save_name: str) -> None:
        super().__init__(trainer)
        self.save_dir = save_dir
        self.save_every_epoch = save_every_epoch
        self.save_name = save_name
        os.makedirs(self.save_dir, exist_ok=True)
        assert self.save_dir is not None and self.save_every_epoch >= 1 and self.save_name is not None, \
            "save_dir and save_every_epoch must be provided"

    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.save_every_epoch == 0:
            epoch = self.trainer.current_epoch + 1
            assert hasattr(self.trainer, 'model'), "trainer must have model attribute"
            path = os.path.join(self.save_dir, f"{self.save_name}_epoch{epoch}.pth")
            torch.save(self.trainer.model.state_dict(), path)
            print(f"Model saved at {path}")


def training():
    parser = argparse.ArgumentParser(description='Fully supervised ResNet34U_f with 10%% labeled (same style as run_full_depthEnhanceTrain).')
    parser.add_argument('--config', type=str, default='cfg/supervised_10pct.yaml', help='Path to YAML config')
    args, unknown = parser.parse_known_args()
    cfg = Config(config_file=args.config, cli_overrides=unknown)

    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))

    val_perc = int(cfg.get('data.val_split_perc', 0))
    resize_h = cfg.get('data.test.resize_height', cfg.get('data.eval.resize_height', 320))
    resize_w = cfg.get('data.test.resize_width', cfg.get('data.eval.resize_width', 320))
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor()
    ])

    train_dataloader = None
    val_dataloader = None

    if val_perc == 0:
        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth', False), list_name=None
        )
        train_num = len(train_data)
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        print(f"Total training images: {train_num}, using labeled: {labeled_num} ({100 * labeled_num / train_num:.1f}%)")
        labeled_subset = Subset(train_data, range(labeled_num))
        batch_size = int(cfg.get('data.labeled_bs', cfg.get('data.batch_size', 4)))
        train_dataloader = torch.utils.data.DataLoader(
            labeled_subset, batch_size=batch_size, shuffle=True, num_workers=cfg.get('data.num_workers', 0)
        )

    elif val_perc < 100:
        print(f"Validation split: {val_perc}%")
        data_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), 'images')
        all_files = np.random.permutation([f for f in os.listdir(data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]).tolist()
        total_num = len(all_files)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        val_files = all_files[:val_num]
        train_files = all_files[val_num:]

        train_data_full = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth', False), list_name=train_files
        )
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        print(f"Train images: {train_num}, labeled: {labeled_num} ({100 * labeled_num / train_num:.1f}%)")
        labeled_subset = Subset(train_data_full, range(labeled_num))
        batch_size = int(cfg.get('data.labeled_bs', cfg.get('data.batch_size', 4)))
        train_dataloader = torch.utils.data.DataLoader(
            labeled_subset, batch_size=batch_size, shuffle=True, num_workers=cfg.get('data.num_workers', 0)
        )

        train_dataset_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'))
        val_data = ImageFolderDataset(
            dataset_root=train_dataset_root,
            image_dirname='images', mask_dirname='masks',
            transform=val_test_transform, list_name=val_files
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.get('data.test.batch_size', 1), shuffle=False, num_workers=0
        )

    else:
        raise ValueError("val_perc must be between 0 and 100.")

    test_data = ImageFolderDataset(
        dataset_root=cfg.get('data.test.dataset_root'),
        image_dirname=cfg.get('data.test.image_dirname'),
        mask_dirname=str(cfg.get('data.test.mask_dirname')),
        transform=val_test_transform, list_name=None
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.get('data.test.batch_size', 1), shuffle=False, num_workers=0
    )

    print(f"Total evaluation images: {len(test_dataloader.dataset)}")
    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    total_iter = cfg.get('total_iter')
    nEpoch = total_iter // iters_per_epoch
    print(f"Total iterations: {total_iter} | Iters/epoch: {iters_per_epoch} => nEpoch: {nEpoch}")

    model_name = cfg.get('model.stu_model.name', cfg.get('model.name', 'ResNet34U_f'))
    num_classes = cfg.get('model.stu_model.num_channels_output', cfg.get('model.num_channels_output', 1))
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

    trainer = SupervisedTrainer(
        model, train_dataloader, optimizer, scheduler, nEpoch
    )

    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        hook_builder(SupervisedEvalHook, eval_data_loader=val_dataloader,
                     eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)), prefix='val_')
    hook_builder(SupervisedEvalHook, eval_data_loader=test_dataloader,
                 eval_every_epoch=int(cfg.get('Hook.SupervisedEvalHook.eval_every_epoch', 2)), prefix='test_')
    hook_builder(FrequentSaveModelSingle, save_dir=cfg.get('Hook.FrequentSaveModelSingle.save_dir', 'save_dir/'),
                 save_every_epoch=int(cfg.get('Hook.FrequentSaveModelSingle.save_every_epoch', 5)),
                 save_name=cfg.get('Hook.FrequentSaveModelSingle.save_name', 'supervised_10pct'))
    hook_builder(LoggerHook, logger_file=cfg.get('Hook.LoggerHook.logger_file', 'logs/supervised.json'))
    hook_builder(MLFlowLoggerHook,
                 dagshub_repo_owner=str(cfg.get('Hook.MLFlowLoggerHook.dagshub_repo_owner', '')),
                 dagshub_repo_name=str(cfg.get('Hook.MLFlowLoggerHook.dagshub_repo_name', '')),
                 experiment_name=cfg.get('Hook.MLFlowLoggerHook.experiment_name', 'supervised_10pct'),
                 dir_save_plot=cfg.get('Hook.MLFlowLoggerHook.dir_save_plot', 'plots'),
                 logging_fields=list(cfg.get('Hook.MLFlowLoggerHook.logging_fields', ['*loss*', '*Dice*', '*IoU*', '*ACC*'])))

    trainer.train()


if __name__ == '__main__':
    training()
    print("Training completed!")
