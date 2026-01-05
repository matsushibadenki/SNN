# ファイルパス: snn_research/training/trainers/self_supervised.py
# Title: Self-Supervised Trainer (自己教師あり学習) - 戻り値修正版
# Description:
#   BreakthroughTrainerを拡張し、TCLなどの自己教師あり学習を行うトレーナー。
#   修正: モデルの戻り値アンパック処理を柔軟に変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple
import logging

from spikingjelly.activation_based import functional  # type: ignore

from .breakthrough import BreakthroughTrainer
from snn_research.training.losses import SelfSupervisedLoss

logger = logging.getLogger(__name__)


class SelfSupervisedTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Union[Tuple[torch.Tensor, ...], Dict[str, Any]], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids: torch.Tensor
        target_ids: torch.Tensor
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids', batch.get(
                'input_images')).to(self.device)  # type: ignore
            target_ids = batch.get('labels').to(self.device)  # type: ignore
        else:
            input_ids, target_ids = [t.to(self.device) for t in batch[:2]]

        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            if self.optimizer is None:
                raise ValueError("Optimizer is not defined")
            with torch.set_grad_enabled(is_train):
                # SelfSupervisedではFull Hiddensを返す
                outputs = self.model(
                    input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=True)

                # --- 修正: 柔軟なアンパック処理 ---
                full_hiddens: torch.Tensor
                spikes: torch.Tensor = torch.tensor(0.0)
                mem: torch.Tensor = torch.tensor(0.0)

                if isinstance(outputs, tuple):
                    full_hiddens = outputs[0]
                    if len(outputs) > 1:
                        spikes = outputs[1]
                    if len(outputs) > 2:
                        mem = outputs[2]
                else:
                    full_hiddens = outputs
                # -------------------------------

                assert isinstance(self.criterion, SelfSupervisedLoss)
                loss_dict = self.criterion(
                    full_hiddens, target_ids, spikes, mem, self.model)

        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                self.optimizer.step()

        loss_dict['accuracy'] = torch.tensor(0.0, device=self.device)
        loss_dict['avg_cutoff_steps'] = torch.tensor(16.0, device=self.device)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
