# snn_research/training/base_trainer.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Protocol, runtime_checkable
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@runtime_checkable
class LoggerProtocol(Protocol):
    """ロギング用インターフェース定義"""
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None: ...

class AbstractTrainer:
    """
    SNN学習のための汎用トレーナー基底クラス。
    """
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        save_dir: Optional[str] = None,
        logger_client: Optional[LoggerProtocol] = None
    ):
        # Improved Device Selection (Supports MPS)
        if device == "cpu":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger_client = logger_client

        if config is not None:
            if isinstance(config, dict):
                self.config = OmegaConf.create(config)
            else:
                self.config = config
        else:
            self.config = OmegaConf.create()

        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            cfg_save_dir = OmegaConf.select(self.config, "training.save_dir", default="workspace/runs/checkpoints")
            self.save_dir = Path(str(cfg_save_dir))

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = -float('inf')

    def save_checkpoint(self, filename: str = "checkpoint.pth", metric: Optional[float] = None) -> None:
        path = self.save_dir / filename
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric
        }
        if self.optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            best_path = self.save_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info(f"New best model saved to {best_path} (metric: {metric:.4f})")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', -float('inf'))

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {path} (Epoch: {self.current_epoch})")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        return self.validate(val_loader)

# --- Backward Compatibility Alias ---
BaseTrainer = AbstractTrainer