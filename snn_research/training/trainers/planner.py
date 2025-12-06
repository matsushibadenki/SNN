# ファイルパス: snn_research/training/trainers/planner.py
# Title: Planner Trainer
# Description: PlannerSNNモデル専用のトレーナー。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from snn_research.training.losses import PlannerLoss

class PlannerTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.train()
        progress_bar = tqdm(dataloader, desc=f"Planner Training Epoch {epoch}")
        for batch in progress_bar:
            if isinstance(batch, dict):
                 input_ids = batch['input_ids'].to(self.device)
                 target_plan = batch['labels'].to(self.device)
            else:
                 input_ids, target_plan = [t.to(self.device) for t in batch]
            
            self.optimizer.zero_grad()
            skill_logits = self.model(input_ids)
            assert isinstance(self.criterion, PlannerLoss)
            loss_dict = self.criterion(skill_logits, target_plan)
            loss: torch.Tensor = loss_dict['total']
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix({"loss": loss.item()})