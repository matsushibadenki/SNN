# ファイルパス: snn_research/training/trainers/stdp.py
# 日本語タイトル: STDP学習トレーナー (No-Grad版)
# 修正: 勾配計算の記録を明示的に禁止し、ポリシーを遵守。

from typing import Dict, cast, Any
import torch
import torch.nn as nn
import logging
from snn_research.training.base_trainer import AbstractTrainer, DataLoader
from snn_research.core.network import AbstractNetwork

logger = logging.getLogger(__name__)


class STDPTrainer(AbstractTrainer):
    """
    STDP (Spike-Timing Dependent Plasticity) に基づくトレーナー。
    誤差逆伝播を使わず、前後のスパイクタイミングのみで重みを更新する。
    """

    def __init__(self, model, learning_rate: float = 0.001, **kwargs):
        super().__init__(model)
        self.learning_rate = learning_rate

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Starting STDP training epoch {self.current_epoch}...")

        total_spikes: float = 0.0
        batch_count = 0

        # ポリシー遵守: 勾配計算は一切行わない
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('input_images'))
                    targets = batch.get('labels')
                else:
                    inputs, targets = batch

                if inputs is None:
                    continue

                metrics: Dict[str, float] = {}

                # Forward実行 (内部でSTDP更新が行われる想定)
                if isinstance(self.model, AbstractNetwork):
                    _ = self.model.forward(inputs)
                    metrics = {}
                elif hasattr(self.model, 'run_learning_step'):
                    metrics = cast(Any, self.model).run_learning_step(
                        inputs, targets)
                elif isinstance(self.model, nn.Module):
                    _ = self.model(inputs)
                    metrics = {}
                else:
                    logger.warning(
                        f"Unknown model type in STDPTrainer: {type(self.model)}")

                total_spikes += metrics.get('spike_count', 0.0)
                batch_count += 1

        self.current_epoch += 1
        return {'mean_spikes': total_spikes / max(1, batch_count)}
