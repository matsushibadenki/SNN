# ファイルパス: snn_research/core/trainer.py
# (修正: AbstractSNNNetwork対応)
# タイトル: 抽象学習トレーナー (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 3) に基づき、
#   ネットワークの訓練および評価ループを抽象化・管理するクラス。
#   AbstractSNNNetwork (SequentialSNNNetworkなど) と互換性を持つように更新。

import logging
from typing import (
    Dict, Any, Optional, Iterable, List, Tuple, Union,
    Mapping,
    Protocol,
    cast
)

import torch
from torch import Tensor
import torch.nn as nn

# P2-2 (抽象ネットワーク) と 新しい AbstractSNNNetwork をインポート
try:
    from .network import AbstractNetwork
except ImportError:
    class AbstractNetwork(nn.Module): # type: ignore[no-redef]
        def forward(self, i: Tensor, t: Optional[Tensor] = None) -> Dict[str, Tensor]: return {}
        def update_model(self, i: Tensor, t: Optional[Tensor], s: Dict[str, Tensor]) -> Dict[str, Tensor]: return {}

try:
    from .networks.abstract_snn_network import AbstractSNNNetwork
except ImportError:
    class AbstractSNNNetwork(nn.Module): # type: ignore[no-redef]
        def forward(self, x: Tensor) -> Tensor: return x
        def run_learning_step(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Any]: return {}
        def reset_state(self) -> None: pass

# データローダーの型エイリアス
Batch = Tuple[Tensor, Tensor]
DataLoader = Iterable[Batch]

# メトリクスの型エイリアス
MetricValue = Union[Tensor, float, int]
MetricsMap = Mapping[str, MetricValue]
MetricsDict = Dict[str, MetricValue]

class LoggerProtocol(Protocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        ...

logger: logging.Logger = logging.getLogger(__name__)

class AbstractTrainer:
    """
    P3-1, P3-3: ネットワークの訓練・評価ループを管理するトレーナー。
    AbstractNetwork (旧) と AbstractSNNNetwork (新) の両方に対応。
    """

    def __init__(
        self, 
        model: Union[AbstractNetwork, AbstractSNNNetwork],
        logger_client: Optional[LoggerProtocol] = None
    ) -> None:
        self.model = model
        self.logger_client = logger_client
        self.current_epoch = 0

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        if logger:
            logger.info(f"Starting training epoch {self.current_epoch}...")
        
        epoch_metrics: List[MetricsMap] = []
        
        for i, batch in enumerate(data_loader):
            inputs: Tensor
            targets: Tensor
            inputs, targets = batch
            
            batch_metrics: MetricsMap = {}

            # --- AbstractSNNNetwork (New) ---
            if isinstance(self.model, AbstractSNNNetwork):
                # 1. Forward pass
                output = self.model(inputs)
                
                # 2. Learning step (STDP / PC)
                batch_metrics = self.model.run_learning_step(inputs, targets)
                
                # 必要に応じてoutputをmetricsに追加
                # batch_metrics['loss'] = ... (もし計算していれば)

            # --- AbstractNetwork (Old) ---
            elif isinstance(self.model, AbstractNetwork):
                model_state = self.model.forward(inputs, targets)
                batch_metrics = self.model.update_model(inputs, targets, model_state)
            
            else:
                # Fallback for generic nn.Module (only forward)
                self.model(inputs)

            epoch_metrics.append(batch_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(epoch_metrics)
        if logger:
            logger.info(f"Training epoch finished. Metrics: {aggregated_metrics}")
            
        if self.logger_client:
            log_data: Dict[str, Any] = {
                f"train/{k}": v for k, v in aggregated_metrics.items()
            }
            self.logger_client.log(log_data, step=self.current_epoch)
            
        self.current_epoch += 1
        return aggregated_metrics

    def evaluate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        if logger:
            logger.info(f"Starting evaluation epoch {self.current_epoch}...")
            
        epoch_eval_metrics: List[MetricsMap] = []

        for batch in data_loader:
            inputs: Tensor
            targets: Tensor
            inputs, targets = batch
            
            model_state: Dict[str, Tensor] = {}
            output: Optional[Tensor] = None

            # --- AbstractSNNNetwork (New) ---
            if isinstance(self.model, AbstractSNNNetwork):
                # Forward only
                output = self.model(inputs)
                model_state['output'] = output
                
            # --- AbstractNetwork (Old) ---
            elif isinstance(self.model, AbstractNetwork):
                model_state = self.model.forward(inputs, targets=targets)
                output = model_state.get('output')
            
            # 評価メトリクスの計算
            eval_metrics = self._calculate_eval_metrics(output, targets)
            epoch_eval_metrics.append(eval_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(epoch_eval_metrics)
        if logger:
            logger.info(f"Evaluation epoch finished. Metrics: {aggregated_metrics}")
            
        if self.logger_client:
            log_data: Dict[str, Any] = {
                f"eval/{k}": v for k, v in aggregated_metrics.items()
            }
            self.logger_client.log(log_data, step=self.current_epoch)
            
        return aggregated_metrics

    def _aggregate_metrics(self, metrics_list: List[MetricsMap]) -> Dict[str, float]:
        if not metrics_list:
            return {}

        aggregated: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        
        for batch_metrics in metrics_list:
            for key, value in batch_metrics.items():
                try:
                    float_val: float
                    if isinstance(value, Tensor):
                        float_val = float(value.item())
                    else:
                        float_val = float(value)
                        
                    aggregated[key] = aggregated.get(key, 0.0) + float_val
                    counts[key] = counts.get(key, 0) + 1
                except (ValueError, TypeError):
                    pass

        for key in aggregated:
            if counts[key] > 0:
                aggregated[key] /= counts[key]
        
        return aggregated

    def _calculate_eval_metrics(self, output: Optional[Tensor], targets: Tensor) -> Dict[str, float]:
        if output is None:
            return {'accuracy': 0.0}
        
        try:
            # output shape: (Batch, Features) or (Batch, Time, Features)
            if output.dim() == 3:
                # 時間平均をとる (Rate coding assumption)
                output = output.mean(dim=1)
            
            predicted = output.argmax(dim=1)
            correct = (predicted == targets)
            accuracy = correct.sum().item() / targets.size(0)
            return {'accuracy': accuracy}
        except Exception:
            return {'accuracy': 0.0}