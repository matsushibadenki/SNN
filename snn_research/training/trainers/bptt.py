# ファイルパス: snn_research/training/trainers/bptt.py
# Title: BPTT Trainer (Robust & Stabilized)
# Description: Backpropagation Through Time (BPTT) を用いたトレーナー。Gradient Clippingと出力形式の柔軟性を追加。

import torch
import torch.nn as nn
from torch.optim import Adam
from omegaconf import DictConfig
from typing import Any, Tuple, Union, Optional, Dict

from snn_research.training.base_trainer import AbstractTrainer
from torch.utils.data import DataLoader


class BPTTTrainer(AbstractTrainer):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        save_dir: Optional[str] = None
    ):
        # Allow passing optimizer, or create default if None
        if optimizer is None:
            lr = config.training.get("learning_rate", 1e-3)
            weight_decay = config.training.get("weight_decay", 0.0)
            optimizer = Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)

        super().__init__(
            model=model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=save_dir
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_grad_norm = config.training.get("max_grad_norm", 1.0)

    def _calculate_loss(self, outputs: Union[torch.Tensor, Tuple[Any, ...]], targets: torch.Tensor) -> torch.Tensor:
        """
        モデルの出力形式に応じてLossを計算する。
        Tuple出力の場合は最初の要素（logits/spikes）を使用する。
        """
        prediction: torch.Tensor
        if isinstance(outputs, tuple):
            prediction = outputs[0]
        else:
            prediction = outputs

        # 形状の整合性確認
        # Prediction: [Batch, Length, Classes] or [Batch, Classes]
        # Targets: [Batch, Length] or [Batch]

        if prediction.dim() == 3 and targets.dim() == 2:
            # Sequence Task: Flatten for CrossEntropy
            return self.criterion(prediction.reshape(-1, prediction.size(-1)), targets.reshape(-1))
        elif prediction.dim() == 3 and targets.dim() == 1:
            # Classification Task with [Batch, Time, Class] output -> Average or Last
            # ここでは単純化のため、時間次元をBatchとして扱うか、時間平均を取るロジックが必要だが、
            # 既存コードに合わせてPermute -> Reshapeを行う
            return self.criterion(prediction.permute(1, 0, 2).reshape(-1, prediction.shape[2]), targets.repeat_interleave(prediction.shape[1]))
        else:
            # Simple Classification
            return self.criterion(prediction, targets)

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()

        # Optimizer checks
        if self.optimizer is None:
            raise ValueError("Optimizer must be set for training")

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(data)

        # Loss calculation
        loss = self._calculate_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (SNNでは必須級)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        loss_val = loss.item()
        del loss  # メモリ解放
        return loss_val

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """1エポック分の学習を実行"""
        total_loss = 0.0
        num_batches = 0

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            loss = self.train_step(data, targets)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """検証ループ"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self._calculate_loss(outputs, targets)
                total_loss += loss.item()

                # Simple accuracy calc (adjust based on output shape)
                prediction: torch.Tensor
                if isinstance(outputs, tuple):
                    prediction = outputs[0]
                else:
                    prediction = outputs

                if prediction.dim() == 3:
                    # Mean over time
                    prediction = prediction.mean(dim=1)

                _, predicted = prediction.max(1)
                total += targets.size(0)
                correct += int(predicted.eq(targets).sum().item())

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        acc = 100. * correct / total if total > 0 else 0.0

        return {"val_loss": avg_loss, "val_acc": acc}
