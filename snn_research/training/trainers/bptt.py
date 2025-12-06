# ファイルパス: snn_research/training/trainers/bptt.py
# Title: BPTT Trainer
# Description: 従来のBackpropagation Through Time (BPTT) を用いたシンプルなトレーナー。

import torch
import torch.nn as nn
from torch.optim import Adam
from omegaconf import DictConfig

class BPTTTrainer:
    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(self.model.parameters(), lr=config.training.get("learning_rate", 1e-3))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = self.config.model.get("type", "simple")
        
    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.model_type == "spiking_transformer": return self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        else: return self.criterion(outputs.permute(1, 0, 2).reshape(-1, outputs.shape[2]), targets.reshape(-1))
        
    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.model(data) if self.model_type != "spiking_transformer" else self.model(data)[0]
        loss = self._calculate_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()