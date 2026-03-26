from engine.Hook import MLFlowLoggerHook
import os
import copy
import typing
import torch
import mlflow
from engine.Trainer import Trainer


class ExtendMLFlowLoggerHook(MLFlowLoggerHook):
    ## combine smartsavehook and mlflowloggerhook
    def __init__(self, trainer: Trainer, 
                local_dir_save_ckpt: str, dagshub_dir_save_ckpt: str, max_save_epoch_interval: int, criteria: str, 
                dagshub_destination_src_file: str = 'src_file', list_src_dir_files: typing.List[str] = [],
                **kwargs) -> None:
        super().__init__(trainer, **kwargs)
        self.local_dir_save_ckpt = local_dir_save_ckpt
        self.dagshub_dir_save_ckpt = dagshub_dir_save_ckpt
        self.max_save_epoch_interval = max_save_epoch_interval
        self.criteria = criteria

        self.dagshub_destination_src_file = dagshub_destination_src_file
        self.list_src_dir_files = list_src_dir_files

    def _log_source_files(self) -> None:  
        for dir_file in self.list_src_dir_files:
            if os.path.isfile(dir_file):
                mlflow.log_artifact(dir_file, artifact_path=self.dagshub_destination_src_file)
                print(f"Logged source file: {dir_file}")
            elif os.path.isdir(dir_file):
                mlflow.log_artifacts(dir_file, artifact_path=os.path.join(self.dagshub_destination_src_file, os.path.basename(dir_file)))
                print(f"Logged source dir: {dir_file}")
            else:
                print(f"[WARN] Source file/dir not found, skipped: {dir_file}")
    def before_train(self) -> None:
        super().before_train()
        self._log_source_files()

    def after_train_epoch(self) -> None:
        super().after_train_epoch()
        latest = self.trainer.info_storage.latest_info()
        self.patience += 1

        if self.criteria not in latest:
            return
        criteria_value = latest[self.criteria]
        if self.best_record is None or criteria_value > self.best_record:
            self.best_record = criteria_value
            self.has_improved = True
            self.ckpt_info['ckpt'] = copy.deepcopy(self.trainer.get_Trainer_ckpt())
            self.ckpt_info['epoch'] = self.trainer.current_epoch + 1

        if self.patience >= self.max_save_epoch_interval and self.has_improved:
            torch.save(self.ckpt_info['ckpt'], os.path.join(self.local_dir_save_ckpt, f"{self.experiment_name}_epoch{self.ckpt_info['epoch']}.pth"))
            self.has_improved = False
            self.patience = 0

    def after_train(self) -> None:
        super().after_train()
        if self.ckpt_info['ckpt'] is not None:
            final_ckpt_save_dir = os.path.join(self.dagshub_dir_save_ckpt, f"final_{self.experiment_name}_epoch{self.ckpt_info['epoch']}.pth")
            torch.save(self.ckpt_info['ckpt'], final_ckpt_save_dir)
            print(f"Final model saved at {final_ckpt_save_dir}")
            mlflow.log_artifact(final_ckpt_save_dir, artifact_path=self.dagshub_dir_save_ckpt)
            print(f"Final model logged to DagsHub MLflow!")

