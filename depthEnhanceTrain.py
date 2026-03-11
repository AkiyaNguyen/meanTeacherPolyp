from engine.Config import Config, HookBuilder
from engine.Trainer import Trainer
from engine.Hook import LoggerHook, EvalHook, HookBase, MLFlowLoggerHook
from torchvision.datasets import ImageFolder
import copy

from test.eval import evaluate, ImageFolderDataset
from data.transform import Resize, ToTensor
from torchvision import transforms
import typing
import argparse

from utils.common import *
from utils.dpa import dpa
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
import optuna

from torch.optim.lr_scheduler import LambdaLR
from utils.ramps import sigmoid_rampup


def softmax_mse_loss(input_logits, target_logits):
    num_classes = input_logits.size()[1]
    if num_classes == 1:
        loss = F.mse_loss(input_logits, target_logits, reduction='mean') / num_classes
    else:
        loss = F.mse_loss(input_logits, target_logits, reduction='mean') / num_classes
    return loss


def dice_loss(pred, target, smooth=1):
    size = pred.size(0)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_score / size


def mask_bce_loss(pred, target, threshold=0.95):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    mask = ((target > threshold) | (target < 1 - threshold)).float()
    pred, target = pred * mask, target * mask
    loss = nn.BCELoss(reduction='mean')
    return loss(pred, target)


def bce_dice_loss(pred, target):
    bce_loss_value = mask_bce_loss(pred, target)
    dice_loss_value = dice_loss(pred, target)
    return bce_loss_value + dice_loss_value


def feature_similarity_loss(rgb_feat, rgbd_feat):
    # Pool to (N, C) so spatial size mismatch (e.g. RGB vs RGB-D branches) is handled
    rgb_feat = F.adaptive_avg_pool2d(rgb_feat, 1).flatten(1)   # (N, C)
    rgbd_feat = F.adaptive_avg_pool2d(rgbd_feat, 1).flatten(1)
    rgb_feat = F.normalize(rgb_feat, dim=1)
    rgbd_feat = F.normalize(rgbd_feat, dim=1)
    sim = F.cosine_similarity(rgb_feat, rgbd_feat, dim=1)
    return 1 - sim.mean()


class DepthEnhance_MT_Trainer(Trainer):
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
        self.class_criterion = nn.BCELoss()
        self.consistency_criterion = softmax_mse_loss
        self.dpa_loss = bce_dice_loss

    def _get_current_consistency_weight(self, global_step):
        return self.consistency * sigmoid_rampup(current=global_step, rampup_length=self.consistency_rampup)

    def _update_ema_variable(self, global_step):
        coeff = min(1 - 1 / (global_step + 1), self.ema_alpha)
        for tea_param, stu_param in zip(self.tea_model.rgb_branch.parameters(), self.stu_model.parameters()):
            tea_param.data.mul_(coeff).add_(stu_param.data, alpha=1 - coeff)

    def get_Trainer_ckpt(self) -> dict:
        result = dict()
        result['stu_model'] = self.stu_model.state_dict()
        result['tea_model'] = self.tea_model.state_dict()
        result['current_epoch'] = self.current_epoch + 1
        result['stu_optimizer'] = self.stu_optimizer.state_dict()
        result['tea_optimizer'] = self.tea_optimizer.state_dict()
        result['scheduler'] = self.scheduler.state_dict()
        result['tea_scheduler'] = self.tea_scheduler.state_dict() if self.tea_scheduler is not None else None
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
        self.tea_model.eval()
        device = next(self.stu_model.parameters()).device

        phase1_info = {'labeled_loss': [], 'unlabeled_rgbd_loss': [],
                       'consistency_weight': [], 'unlabeled_rgbd_cutmix_loss': [], 'loss': []}
        phase2_info = {'teacher_labeled_loss': []}

        # ========== PHASE 1: Train Student + EMA Update ==========
        for batch_id, data in enumerate(self.train_dataloader):
            self.stu_optimizer.zero_grad()
            img_s, img, label, depth = data['image_s'], data['image'], data['label'], data['depth']
            img_s, img, label, depth = img_s.to(device), img.to(device), label.to(device), depth.to(device)

            unlabeled_img = img[self.labeled_bs:]
            unlabeled_depth = depth[self.labeled_bs:]
            unlabeled_img_s = img_s[self.labeled_bs:]

            label = label[:self.labeled_bs]

            stu_pred = self.stu_model(img_s)
            labeled_stu = stu_pred[:self.labeled_bs]
            unlabeled_stu = stu_pred[self.labeled_bs:]

            with torch.no_grad():
                tea_output = self.tea_model(unlabeled_img, unlabeled_depth)


            unlabeled_img_s_cutmix, ema_pred_u_cutmix = dpa(
                unlabeled_depth, unlabeled_img_s, tea_output, beta=0.3, t=self.current_epoch, T=self.num_epochs
            )
            pred_u_cutmix = self.stu_model(unlabeled_img_s_cutmix)
            loss_consist_rgbd_cutmix = self.dpa_loss(pred_u_cutmix, ema_pred_u_cutmix)

            loss_sup = self.class_criterion(labeled_stu, label)
            loss_consist_rgbd = self.consistency_criterion(unlabeled_stu, tea_output)
            # TODO: add feature similarity loss

            consistency_weight = self._get_current_consistency_weight(
                global_step=batch_id + self.current_epoch * len(self.train_dataloader)
            )
            total_loss = loss_sup + consistency_weight * loss_consist_rgbd + loss_consist_rgbd_cutmix

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), max_norm=1.0)
            self.stu_optimizer.step()
            self._update_ema_variable(global_step=batch_id + self.current_epoch * len(self.train_dataloader))

            phase1_info['labeled_loss'].append(loss_sup.item())
            phase1_info['unlabeled_rgbd_loss'].append(loss_consist_rgbd.item())
            phase1_info['unlabeled_rgbd_cutmix_loss'].append(loss_consist_rgbd_cutmix.item())
            phase1_info['consistency_weight'].append(consistency_weight)
            phase1_info['loss'].append(total_loss.item())
            self.scheduler.step()

        self._add_info({f'phase1_{k}': np.mean(v) for k, v in phase1_info.items()})

        # ========== PHASE 2: Train Teacher Depth/Fusion/Decoder ==========
        self.tea_model.train()
        for batch_id, data in enumerate(self.train_dataloader):
            self.tea_optimizer.zero_grad()
            img_s, img, label, depth = data['image_s'], data['image'], data['label'], data['depth']
            img_s, img, label, depth = img_s.to(device), img.to(device), label.to(device), depth.to(device)

            labeled_depth = depth[:self.labeled_bs]
            labeled_img = img[:self.labeled_bs]
            label = label[:self.labeled_bs]


            for param in self.tea_model.rgb_branch.parameters():
                param.requires_grad_(False)
            tea_labeled_rgbd_output = self.tea_model(labeled_img, labeled_depth)
            # tea_labeled_rgbd_output = tea_labeled_output

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


class StopTrainAtEpoch(HookBase):
    def __init__(self, trainer: Trainer, stop_at_epoch: int) -> None:
        super().__init__(trainer)
        self.stop_at_epoch = stop_at_epoch
    def after_train_epoch(self) -> None:
        if self.trainer.current_epoch + 1 >= self.stop_at_epoch:
            self.trainer.stop_training()
            print(f"Training stopped at epoch {self.stop_at_epoch}")
    

class MeanTeacherEvalHook(EvalHook):
    def __init__(self, trainer: Trainer, eval_data_loader: torch.utils.data.DataLoader, eval_every_epoch: int, prefix: str = '') -> None:
        super().__init__(trainer, eval_data_loader)
        self.eval_every_epoch = eval_every_epoch
        self.prefix = prefix 
        assert self.eval_every_epoch >= 1, "eval_every_epoch must be at least 1"
    def _run_validation(self) -> dict[typing.Any, typing.Any]:
        assert hasattr(self.trainer, 'stu_model') and hasattr(self.trainer, 'tea_model'),\
                 "trainer must have stu_model and tea_model attributes"
        self.trainer.stu_model.eval() # type: ignore
        device = next(self.trainer.stu_model.parameters()).device # type: ignore
        metrics = {
            'stu_ACC_overall': [],
            'tea_rgb_ACC_overall': [],
            'tea_rgbd_ACC_overall': [],
            'stu_Dice': [],
            'tea_rgb_Dice': [],
            'tea_rgbd_Dice': [],
            'stu_IoU': [],
            'tea_rgb_IoU': [],
            'tea_rgbd_IoU': [],
        }
        with torch.no_grad():
            for data in self.eval_data_loader:
                img = data['image'].to(device)
                gt = data['mask'].to(device)
                depth = data['depth'].to(device)
                stu_output = self.trainer.stu_model(img) # type:ignore
                cur_stu_metrics = evaluate(stu_output, gt)

                tea_output = self.trainer.tea_model(img, depth)
                tea_rgb_output, tea_rgbd_output = tea_output['rgb'], tea_output['rgb_depth']

                cur_tea_rgb_metrics = evaluate(tea_rgb_output, gt)
                cur_tea_rgbd_metrics = evaluate(tea_rgbd_output, gt)

                for key, value in cur_stu_metrics.items():
                    metrics['stu_' + key].append(value)
                for key, value in cur_tea_rgb_metrics.items():
                    metrics['tea_rgb_' + key].append(value)
                for key, value in cur_tea_rgbd_metrics.items():
                    metrics['tea_rgbd_' + key].append(value)
        
        return {self.prefix + key: np.mean(value) for key, value in metrics.items()}
    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.eval_every_epoch == 0:
            result = self._run_validation()
            self.trainer._add_info(result)


class FrequentSaveModel(HookBase):   
    def __init__(self, trainer: Trainer, save_dir: str, save_every_epoch: int, save_name: str) -> None:
        super().__init__(trainer)
        self.save_dir = save_dir
        self.save_every_epoch = save_every_epoch
        self.save_name = save_name
        os.makedirs(self.save_dir, exist_ok=True)
        assert self.save_dir is not None and self.save_every_epoch >= 1 and self.save_name is not None, \
            "save_dir and save_every_epoch must be provided when save_model is True"
    def after_train_epoch(self) -> None:
        if (self.trainer.current_epoch + 1) % self.save_every_epoch == 0:
            epoch = self.trainer.current_epoch + 1
            assert hasattr(self.trainer, 'stu_model') and hasattr(self.trainer, 'tea_model'),\
                 "trainer must have stu_model and tea_model attributes"
            # Primary checkpoint = student (same model as val_Dice during training); also save teacher
            stu_path = os.path.join(self.save_dir, f"{self.save_name}_epoch{epoch}.pth")
            tea_path = os.path.join(self.save_dir, f"{self.save_name}_teacher_epoch{epoch}.pth")
            torch.save(self.trainer.stu_model.state_dict(), stu_path) # type: ignore
            torch.save(self.trainer.tea_model.state_dict(), tea_path) # type: ignore
            print(f"Student saved at {stu_path}")
            print(f"Teacher saved at {tea_path}")


def training(trial):
    parser = argparse.ArgumentParser(description='Depth Enhanced RGB-D Polyp Segmentation with Mean Teacher training (argparse for --config; key=value for OmegaConf overrides).')
    parser.add_argument('--config', type=str, default='cfg/depth_enhance_mt.yaml', help='Path to YAML config')
    args, unknown = parser.parse_known_args()
    # unknown contains key=value overrides for OmegaConf (e.g. data.root=..., Trainer.consistency_rampup=200.0)
    cfg = Config(config_file=args.config, cli_overrides=unknown)

    
    ## ================ optimizer sweeping ==========================
    sweep_dict = {}
    sweep_dict['Trainer.fea_sim_weight'] = trial.suggest_float('fea_sim_weight', 0.0, 0.5)
    # sweep_dict['optimizer.lr'] = trial.suggest_float('learning_rate',0.0001, 0.001)
    # sweep_dict['total_iter'] = trial.suggest_int('total_iterations',1000, 1500)
    # sweep_dict['Trainer.consistency_rampup'] = trial.suggest_float('consistency_rampup', 2000, 5000)
    # sweep_dict['Trainer.consistency'] = trial.suggest_float('unsupevised_weight', 2.0, 4.0)

    for key, value in sweep_dict.items():
        cfg.set(key, value)
        
    print(cfg.all_config())
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))


    val_perc = int(cfg.get('data.val_split_perc', 0))

    resize_h = cfg.get('data.eval.resize_height', 320)
    resize_w = cfg.get('data.eval.resize_width', 320)
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor()
    ])


    ## build train, val dataset
    train_dataloader, val_dataloader = None, None
    if val_perc == 0:
        data_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), 'images')
        train_data = getattr(dataset, cfg.get('data.dataset'))(root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'), \
            mode='train', require_depth=cfg.get('data.require_depth'), list_name=None)
        train_num = len(train_data)
        print(f"Total training images: {train_num}")
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        print(f"Labelled images: {labeled_num}")
        print(f"Unlabelled images: {train_num - labeled_num}")
        batch_sampler = TwoStreamBatchSampler(train_num, labeled_num, \
            int(cfg.get('data.labeled_bs')), int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')))
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler, \
                                                       shuffle=cfg.get('data.shuffle'), \
                                                        num_workers=cfg.get('data.num_workers'))

    elif val_perc < 100:
        print(f"Validation split: {val_perc}%")
        data_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), 'images')
        all_files = np.random.permutation([f for f in os.listdir(data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]).tolist()
        total_num = len(all_files)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        print(f"Total training images: {train_num}, validation images: {val_num}")
        val_files = all_files[:val_num]
        train_files = all_files[val_num:]

        train_data = getattr(dataset, cfg.get('data.dataset'))(root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'), \
            mode='train', require_depth=cfg.get('data.require_depth'), list_name=train_files)
        
        if cfg.get('data.label_mode') == 'percentage':
            labeled_num = round(train_num * cfg.get('data.labeled_perc') / 100)
            if labeled_num % 2 != 0:
                labeled_num -= 1
        else:
            labeled_num = cfg.get('data.labeled_num')
        ## labeled_num = 144
        print(f"Total training images: {train_num}, labelled: {labeled_num} ({labeled_num / train_num * 100:.2f}%)")
        batch_sampler = TwoStreamBatchSampler(train_num, labeled_num, \
            int(cfg.get('data.labeled_bs')), int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')))
        
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler, \
                                                       shuffle=cfg.get('data.shuffle'), \
                                                        num_workers=cfg.get('data.num_workers'))
                                                        
        # Val split uses training data root (same as train), restricted by list_name
        train_dataset_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'))
        val_data = ImageFolderDataset(dataset_root=train_dataset_root, \
                image_dirname='images', mask_dirname='masks', depth_dirname='depth-v1', \
                transform=val_test_transform, list_name=val_files)
    
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=cfg.get('data.test.batch_size'), \
                                                     shuffle=False, num_workers=0)

    else:
        raise ValueError("val_perc must be between 0 and 100.")
    

    test_data = ImageFolderDataset(dataset_root=cfg.get('data.test.dataset_root'), image_dirname=cfg.get('data.test.image_dirname'), \
                        mask_dirname=str(cfg.get('data.test.mask_dirname')), \
                        depth_dirname=str(cfg.get('data.test.depth_dirname')), \
                        transform=val_test_transform, list_name=None)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg.get('data.test.batch_size'), \
                                                     shuffle=False, num_workers=0)

    print(f"Total evaluation images: {len(test_dataloader)}")
    # === Compute nEpoch from total_iter ===
    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    nEpoch = cfg.get('total_iter') // iters_per_epoch
    total_iter = cfg.get('total_iter')
    print(f"Total iterations: {total_iter} | Iters/epoch: {iters_per_epoch} => nEpoch: {nEpoch}")

    
    # model = generate_model(cfg).to(device)
    # ema_model = generate_model(cfg, ema=True).to(device)
    stu_model = getattr(models, cfg.get('model.stu_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)
    tea_model = getattr(models, cfg.get('model.tea_model.name'))(num_classes=cfg.get('model.num_channels_output')).to(device)
    
    optimizer = torch.optim.SGD(stu_model.parameters(), lr=cfg.get('optimizer.lr'), \
                                momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))
    tea_depth_branch_optimizer = torch.optim.SGD(tea_model.parameters(), lr=cfg.get('optimizer.lr'), \
                                momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))

    # Student: step() every batch -> lambda gets step index; decay over total_iter.
    scheduler_power = float(cfg.get('scheduler.power'))
    scheduler = LambdaLR(optimizer, lambda e: max(0.0, 1.0 - pow(min(e, total_iter) / total_iter, scheduler_power)))
    # Teacher: step() once per epoch -> lambda gets epoch index; decay over nEpoch.
    tea_depth_branch_scheduler = LambdaLR(tea_depth_branch_optimizer, lambda e: max(0.0, 1.0 - pow(min(e, nEpoch) / nEpoch, scheduler_power)))

    trainer = DepthEnhance_MT_Trainer(
        stu_model, tea_model, train_dataloader,
        optimizer, tea_depth_branch_optimizer,
        scheduler, nEpoch,
        ema_alpha=float(cfg.get('Trainer.ema_decay', 0.999)),
        consistency_rampup=float(cfg.get('Trainer.consistency_rampup')),
        consistency=float(cfg.get('Trainer.consistency')),
        fea_sim_weight=float(cfg.get('Trainer.fea_sim_weight', 0.5)),
        tea_scheduler=tea_depth_branch_scheduler,
    )
     
    hook_builder = HookBuilder(cfg, trainer)
    if val_dataloader is not None:
        val_hook = hook_builder(MeanTeacherEvalHook, eval_data_loader=val_dataloader, \
                eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='val_')
    else:
        val_hook = None
    test_hook = hook_builder(MeanTeacherEvalHook, eval_data_loader=test_dataloader, \
                 eval_every_epoch=int(cfg.get('Hook.MeanTeacherEvalHook.eval_every_epoch')), prefix='test_')
        
    hook_builder(FrequentSaveModel, save_dir=cfg.get('Hook.FrequentSaveModel.save_dir'), \
                save_every_epoch=int(cfg.get('Hook.FrequentSaveModel.save_every_epoch')), \
            save_name=cfg.get('Hook.FrequentSaveModel.save_name'))
    hook_builder(LoggerHook, logger_file='logs/simple.json')
    hook_builder(MLFlowLoggerHook, dagshub_repo_owner=str(cfg.get('Hook.MLFlowLoggerHook.dagshub_repo_owner')), \
                dagshub_repo_name=str(cfg.get('Hook.MLFlowLoggerHook.dagshub_repo_name')), \
                experiment_name=cfg.get('Hook.MLFlowLoggerHook.experiment_name'), \
                dir_save_plot=cfg.get('Hook.MLFlowLoggerHook.dir_save_plot'), \
                logging_fields=list(cfg.get('Hook.MLFlowLoggerHook.logging_fields')))
    hook_builder(StopTrainAtEpoch, stop_at_epoch=int(cfg.get('Hook.StopTrainAtEpoch.stop_at_epoch')))


    trainer.train()

    ## add this for optuna tuning
    criteria = 'val_tea_rgb_Dice' if val_hook is not None else 'test_tea_rgb_Dice'
    for info in reversed(trainer.info_storage.info_storage):
        if criteria in info:
            return info[criteria]
    raise ValueError(f"Criteria {criteria} does not exist in info_storage")

if __name__ == '__main__':
    
    study = optuna.create_study(direction='maximize')
    study.optimize(training, n_trials=4)
    
    