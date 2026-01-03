# ファイルパス: snn_research/training/trainers/bptt.py
# Title: BPTT Trainer (Robust & Stabilized)
# Description: Backpropagation Through Time (BPTT) を用いたトレーナー。Gradient Clippingと出力形式の柔軟性を追加。

import torch
import torch.nn as nn
from torch.optim import Adam
from omegaconf import DictConfig
from typing import Any, Tuple, Union

class BPTTTrainer:
    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config
        
        lr = config.training.get("learning_rate", 1e-3)
        weight_decay = config.training.get("weight_decay", 0.0)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = self.config.model.get("type", "simple")
        self.max_grad_norm = config.training.get("max_grad_norm", 1.0) # Gradient Clipping設定

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
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(data)
        
        # Loss calculation
        loss = self._calculate_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping (SNNでは必須級)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        loss_val = loss.item()
        del loss # メモリ解放
        return loss_val