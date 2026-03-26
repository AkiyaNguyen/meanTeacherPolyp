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
                meta_info: dict | None = None, dagshub_meta_dir: str = 'meta',
                local_dir_save_ckpt: str = 'ckpt', dagshub_dir_save_ckpt: str = 'ckpt', max_save_epoch_interval: int = 50, criteria: str = 'test_stu_Dice',
                dagshub_destination_src_file: str = 'src_file', list_src_dir_files: typing.List[str] | None = None,
                **kwargs) -> None:
        super().__init__(trainer, **kwargs)
        self.meta_info = meta_info if meta_info is not None else {}
        self.dagshub_meta_dir = dagshub_meta_dir
        self.local_dir_save_ckpt = local_dir_save_ckpt
        self.dagshub_dir_save_ckpt = dagshub_dir_save_ckpt
        self.max_save_epoch_interval = max_save_epoch_interval
        self.criteria = criteria

        self.dagshub_destination_src_file = dagshub_destination_src_file
        self.list_src_dir_files = list(list_src_dir_files) if list_src_dir_files is not None else []

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
    def _log_meta_info(self) -> None:
        if self.meta_info:
            # MLflow API: second arg is artifact_file (path under run artifacts), not artifact_path
            mlflow.log_dict(self.meta_info, artifact_file=os.path.join(self.dagshub_meta_dir, 'meta_info.json'))
            print(f"Logged meta info to MLflow!")
    def before_train(self) -> None:
        super().before_train()
        self._log_source_files()
        self._log_meta_info()

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

