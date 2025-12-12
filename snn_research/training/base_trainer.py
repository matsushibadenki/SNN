# ファイルパス: snn_research/training/base_trainer.py
# 日本語タイトル: 抽象学習トレーナー (修正版)
# 機能説明: 
#   ネットワークの訓練・評価ループを管理するトレーナー。
#   
#   修正点:
#   - インポートパスを整理し、AbstractNetwork (core/network.py) と 
#     AbstractSNNNetwork (core/networks/abstract_snn_network.py) の両方を正しく扱えるように修正。

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Iterable, List, Tuple, Union, Mapping, Protocol

import torch
from torch import Tensor
import torch.nn as nn

# 絶対インポートによる正しい基底クラスの参照
from snn_research.core.network import AbstractNetwork
# AbstractSNNNetworkはまだアップロードされたファイル群には含まれていない場合がありますが、
# 存在する場合の正しいパスを指定し、なければダミーではなくImportErrorを許容する設計にします。
try:
    from snn_research.core.networks.abstract_snn_network import AbstractSNNNetwork
except ImportError:
    # AbstractSNNNetworkが未実装の場合のフォールバック（型チェック用）
    class AbstractSNNNetwork(nn.Module): # type: ignore
        def run_learning_step(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Any]:
            return {}
        def reset_state(self) -> None: pass

# データローダーの型エイリアス
Batch = Tuple[Tensor, Tensor]
DataLoader = Iterable[Batch]

# メトリクスの型エイリアス
MetricValue = Union[Tensor, float, int]
MetricsMap = Mapping[str, MetricValue]

class LoggerProtocol(Protocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        ...

logger = logging.getLogger(__name__)

class AbstractTrainer:
    """
    ネットワークの訓練・評価ループを管理するトレーナー。
    AbstractNetwork (P2系) と AbstractSNNNetwork (P4系) の両方に対応。
    """

    def __init__(
        self, 
        model: Union[AbstractNetwork, AbstractSNNNetwork, nn.Module],
        logger_client: Optional[LoggerProtocol] = None
    ) -> None:
        self.model = model
        self.logger_client = logger_client
        self.current_epoch = 0

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Starting training epoch {self.current_epoch}...")
        
        epoch_metrics: List[MetricsMap] = []
        
        for i, batch in enumerate(data_loader):
            if isinstance(batch, dict):
                 inputs = batch.get('input_ids', batch.get('input_images')) # type: ignore
                 targets = batch.get('labels') # type: ignore
            else:
                 inputs, targets = batch
            
            if inputs is None: continue

            batch_metrics: MetricsMap = {}

            # --- AbstractSNNNetwork (New Interface) ---
            if isinstance(self.model, AbstractSNNNetwork):
                # 1. Forward pass
                _ = self.model(inputs)
                # 2. Learning step (STDP / PC)
                batch_metrics = self.model.run_learning_step(inputs, targets)

            # --- AbstractNetwork (Old Interface) ---
            elif isinstance(self.model, AbstractNetwork):
                model_state = self.model.forward(inputs, targets)
                batch_metrics = self.model.update_model(inputs, targets, model_state)
            
            # --- Standard nn.Module (Fallback) ---
            else:
                # 学習ロジックを持たないモデルの場合は何もしないか、
                # 外部オプティマイザに依存する（このクラスは局所学習用）
                _ = self.model(inputs) # type: ignore

            epoch_metrics.append(batch_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(epoch_metrics)
        logger.info(f"Training epoch finished. Metrics: {aggregated_metrics}")
            
        if self.logger_client:
            log_data: Dict[str, Any] = {
                f"train/{k}": v for k, v in aggregated_metrics.items()
            }
            self.logger_client.log(log_data, step=self.current_epoch)
            
        self.current_epoch += 1
        return aggregated_metrics

    def evaluate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Starting evaluation epoch {self.current_epoch}...")
            
        epoch_eval_metrics: List[MetricsMap] = []

        for batch in data_loader:
            if isinstance(batch, dict):
                 inputs = batch.get('input_ids', batch.get('input_images')) # type: ignore
                 targets = batch.get('labels') # type: ignore
            else:
                 inputs, targets = batch
            
            if inputs is None or targets is None: continue
            
            output: Optional[Tensor] = None

            # --- AbstractSNNNetwork ---
            if isinstance(self.model, AbstractSNNNetwork):
                self.model.reset_state()
                output = self.model(inputs) # type: ignore
                
            # --- AbstractNetwork ---
            elif isinstance(self.model, AbstractNetwork):
                # AbstractNetworkのforwardがdictを返すと仮定
                res = self.model.forward(inputs, targets=targets)
                output = res.get('output')
            
            # --- Standard nn.Module ---
            else:
                output = self.model(inputs) # type: ignore
            
            # 評価メトリクスの計算
            if output is not None:
                eval_metrics = self._calculate_eval_metrics(output, targets)
                epoch_eval_metrics.append(eval_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(epoch_eval_metrics)
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

    def _calculate_eval_metrics(self, output: Tensor, targets: Tensor) -> Dict[str, float]:
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