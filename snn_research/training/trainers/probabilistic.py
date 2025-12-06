# ファイルパス: snn_research/training/trainers/probabilistic.py
# Title: Probabilistic Ensemble Trainer (戻り値修正版)
# Description:
#   モンテカルロドロップアウトやSNNの確率性を利用したアンサンブル学習トレーナー。
#   修正: モデルの戻り値アンパック処理を柔軟に変更。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, List
import logging

from spikingjelly.activation_based import functional # type: ignore

from .breakthrough import BreakthroughTrainer
from snn_research.training.losses import ProbabilisticEnsembleLoss

logger = logging.getLogger(__name__)

class ProbabilisticEnsembleTrainer(BreakthroughTrainer):
    def __init__(self, ensemble_size: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self.ensemble_size = ensemble_size
    
    def _run_step(self, batch: Union[Tuple[torch.Tensor, ...], Dict[str, Any]], is_train: bool) -> Dict[str, Any]:
        if is_train: self.model.train()
        else: self.model.eval()
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        input_ids: torch.Tensor
        target_ids: torch.Tensor
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids', batch.get('input_images')).to(self.device) # type: ignore
            target_ids = batch.get('labels').to(self.device) # type: ignore
        else:
            input_ids, target_ids = [t.to(self.device) for t in batch[:2]]

        ensemble_logits: List[torch.Tensor] = []
        for _ in range(self.ensemble_size):
            functional.reset_net(model_to_reset)
            with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                    
                    # --- 修正: 柔軟なアンパック ---
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        # 必要なら spikes, mem 等も取得可能
                    else:
                        logits = outputs
                    
                    ensemble_logits.append(logits)
                    
        ensemble_logits_tensor = torch.stack(ensemble_logits)
        assert isinstance(self.criterion, ProbabilisticEnsembleLoss)
        
        # Loss計算には平均化前のロジットテンソルを渡す
        loss_dict = self.criterion(ensemble_logits_tensor, target_ids, torch.tensor(0.0), torch.tensor(0.0), self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                self.optimizer.step()
                
        with torch.no_grad():
            mean_logits = ensemble_logits_tensor.mean(dim=0)
            preds = torch.argmax(mean_logits, dim=-1)
            accuracy = (preds == target_ids).float().mean()
            loss_dict['accuracy'] = accuracy
            
        loss_dict['avg_cutoff_steps'] = torch.tensor(16.0, device=self.device)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
