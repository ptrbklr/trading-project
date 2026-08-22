import os
import json
from pathlib import Path
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def create_scheduler(optimizer, cfg_sched):
    if cfg_sched.type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg_sched.factor,
            patience=cfg_sched.patience,
        )
    return None

class EarlyStopping:
    def __init__(self, cfg):
        self.enabled = cfg.enabled
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta
        self.best = None
        self.counter = 0

    def step(self, val_loss):
        if not self.enabled:
            return False
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class ModelCheckpoint:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_dir = str(_resolve_project_path(Path(cfg.artifacts.base_dir) / cfg.experiment_name))
        os.makedirs(self.base_dir, exist_ok=True)
        self.best_loss = None

    def _save(self, epoch, model, scalers, metrics, val_loss, tag):
        path = os.path.join(self.base_dir, f"epoch_{epoch}_{tag}")
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pth"))
        torch.save(scalers, os.path.join(path, "scalers.pt"))
        meta = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "metrics": metrics,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def maybe_save(self, epoch, model, scalers, metrics, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save(epoch, model, scalers, metrics, val_loss, "best")
        elif epoch % self.cfg.artifacts.save_every_n_epochs == 0:
            self._save(epoch, model, scalers, metrics, val_loss, "periodic")

class TensorBoardLogger:
    def __init__(self, cfg):
        log_dir = str(_resolve_project_path(Path("logs") / cfg.experiment_name))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log(self, epoch, train_loss, val_loss, metrics, optimizer):
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("metrics/mae", metrics['mae'], epoch)
        self.writer.add_scalar("metrics/mse", metrics['mse'], epoch)
        self.writer.add_scalar("metrics/directional_accuracy", metrics['directional_accuracy'], epoch)
        lr = optimizer.param_groups[0]['lr']
        self.writer.add_scalar("lr", lr, epoch)
