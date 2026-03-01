from engine.Config import Config
from engine.Trainer import Trainer

# import engine


from utils.common import *
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR

from utils.ramps import sigmoid_rampup

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


# def symmetric_mse_loss(input1, input2):

#     assert input1.size() == input2.size()
#     num_classes = input1.size()[1]
#     return torch.sum((input1 - input2)**2) / num_classes



class SimpleMeanTeacherTrainer(Trainer):
    def __init__(self, stu_model, tea_model, train_dataloader, optimizer, scheduler, num_epochs, ema_alpha, \
                    consistency_rampup, consistency, **kwargs) -> None:
        super().__init__(num_epochs, **kwargs)
        self.stu_model = stu_model
        self.tea_model = tea_model
        self.train_dataloader = train_dataloader
        ## student's optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema_alpha = ema_alpha
        self.labeled_bs = self.train_dataloader.batch_sampler.primary_batch_size
        
        self.consistency_rampup = consistency_rampup
        self.consistency = consistency
        # self.ema_decay = ema_decay
        # self.weight_decay = weight_decay

        self.consistency_criterion = softmax_mse_loss

        self.class_criterion = nn.BCEWithLogitsLoss()
        self.save_model = False

    def set_save(self, save_dir: str, save_name: str, save_every_epoch: int):
        self.save_model = True
        self.save_dir = save_dir
        print("save_dir:", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_name = save_name
        self.save_every_epoch = save_every_epoch
        assert self.save_dir is not None and self.save_name is not None and self.save_every_epoch >= 1, \
            "save_dir and save_name must be provided when save_model is True"

    def _update_ema_variable(self, global_step):
        coeff = min(1 - 1 / (global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.tea_model.parameters(), self.stu_model.parameters()):
            ema_param.data.mul_(coeff).add_(1 - coeff, param.data)

    def _start_train_mode(self) -> None:
        self.stu_model.train()
        self.tea_model.eval()
        
    def _get_current_consistency_weight(self, global_step):
        return self.consistency * sigmoid_rampup(current=global_step, \
                                                    rampup_length=self.consistency_rampup)

    def run_step_(self) -> None:
        device = next(self.stu_model.parameters()).device
        loss = {}
        for id, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            # img_s, img, label, depth = data['image_s'], data['image'], data['label'], data['depth']
            img_s, img, label = data['image_s'], data['image'], data['label']
            
            # img_s, img, label, depth = img_s.to(device), img.to(device), label.to(device), depth.to(device)
            img_s, img, label = img_s.to(device), img.to(device), label.to(device)
            
            label = label[:self.labeled_bs]
            # labeled_img_s = img_s[:self.labeled_bs]
            # unlabeled_img_s = img_s[self.labeled_bs:]

            labeled_img = img[:self.labeled_bs]
            unlabeled_img = img[self.labeled_bs:]

            # labeled_depth = depth[:self.labeled_bs]
            # unlabeled_depth = depth[self.labeled_bs:]

            output_s = self.stu_model(img_s)
            labeled_output_s = output_s[:self.labeled_bs]
            unlabeled_output_s = output_s[self.labeled_bs:]
            loss['labeled_loss'] = self.class_criterion(labeled_output_s, label)

            with torch.no_grad():
                teacher_output = self.tea_model(unlabeled_img)
                teacher_output = teacher_output.to(device)
            loss['unlabeled_loss'] = self.consistency_criterion(unlabeled_output_s, teacher_output)
            consistency_weight = self._get_current_consistency_weight(global_step=id + self.current_epoch * len(self.train_dataloader), 
                                                                      )
            loss['loss'] = loss['labeled_loss'] + consistency_weight * loss['unlabeled_loss']

            ## add info for logging.
            self._add_info(loss)

            loss['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

            self._update_ema_variable(global_step=id + self.current_epoch * len(self.train_dataloader))

        if self.save_model:
            if (self.current_epoch + 1) % self.save_every_epoch == 0:
                save_path = os.path.join(self.save_dir, f"{self.save_name}_epoch{self.current_epoch + 1}.pth")
                torch.save(self.stu_model.state_dict(), save_path)
                print(f"Model saved at {save_path}")
    
if __name__ == '__main__':
    cfg = Config(config_file='cfg/simple.yaml')
    
    device = get_proper_device(cfg.get('device'))
    set_seed(cfg.get('seed'))
    
    train_data = getattr(dataset, cfg.get('data.dataset'))(root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'), \
        mode='train', require_depth=cfg.get('data.require_depth'))
    total_num = len(train_data) ## 1450

    if cfg.get('data.label_mode') == 'percentage':
        labeled_num = round(total_num * cfg.get('data.labeled_perc') / 100)
        if labeled_num % 2 != 0:
            labeled_num -= 1
    else:
        labeled_num = cfg.get('data.labeled_num')
    ## labeled_num = 144
    print(f"Total training images: {total_num}, labelled: {labeled_num} ({labeled_num / total_num * 100:.2f}%)")

    batch_sampler = TwoStreamBatchSampler(total_num, labeled_num, \
        cfg.get('data.labeled_bs'), cfg.get('data.batch_size') - cfg.get('data.labeled_bs'))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler, shuffle=cfg.get('data.shuffle'), num_workers=cfg.get('data.num_workers'))

    # === Compute nEpoch from total_iter ===
    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    nEpoch = cfg.get('total_iter') // iters_per_epoch
    print(f"Total iterations: {cfg.get('total_iter')} | Iters/epoch: {iters_per_epoch} => nEpoch: {nEpoch}")

    
    model = generate_model(cfg).to(device)
    ema_model = generate_model(cfg, ema=True).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.get('optimizer.lr'), momentum=cfg.get('optimizer.momentum'), weight_decay=cfg.get('optimizer.weight_decay'))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('scheduler.step_size'), gamma=cfg.get('scheduler.gamma'))
    scheduler = LambdaLR(optimizer, lambda e: 1.0 - pow((e / nEpoch), float(cfg.get('scheduler.power'))))

    trainer = SimpleMeanTeacherTrainer(model, ema_model, train_dataloader, optimizer, scheduler, \
                                       nEpoch, ema_alpha=float(cfg.get('Trainer.ema_decay')), \
                                       consistency_rampup=float(cfg.get('Trainer.consistency_rampup')), \
                                        consistency=float(cfg.get('Trainer.consistency')))
    # print("save dir = ", cfg.get('save.save_dir'))   
    trainer.set_save(save_dir=cfg.get('Trainer.save.save_dir'), save_name=cfg.get('Trainer.save.save_name'), \
                    save_every_epoch=cfg.get('Trainer.save.save_every_epoch'))
    trainer.train()