# matsushibadenki/snn4/snn_research/benchmark/metrics.py
# ベンチマーク評価用のメトリクス関数
#

"""
snn_research/benchmark/metrics.py
評価指標（メトリクス）の定義とレジストリ管理を行うモジュール
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

class MetricRegistry:
    """
    評価指標を一元管理するレジストリクラス。
    """
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """
        メトリクス関数を登録するためのデコレータ
        Args:
            name (str): メトリクス名
        """
        def decorator(func: Callable):
            if name in cls._registry:
                logger.warning(f"Metric '{name}' is already registered. Overwriting.")
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """
        名前からメトリクス関数を取得
        """
        return cls._registry.get(name)

    @classmethod
    def evaluate(cls, metric_name: str, predictions: Any, targets: Any) -> float:
        """
        登録されたメトリクスを使用して評価を実行するヘルパーメソッド
        """
        metric_func = cls.get(metric_name)
        if metric_func is None:
            raise ValueError(f"Metric '{metric_name}' not found in registry.")
        return metric_func(predictions, targets)

# --- 標準的なメトリクスの実装 ---

@MetricRegistry.register("accuracy")
def calculate_accuracy(predictions: Union[torch.Tensor, np.ndarray], 
                      targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    正解率 (Accuracy) を計算
    """
    # Tensor -> Numpy変換
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # データが空の場合
    if len(targets) == 0:
        return 0.0

    # 予測値の形状処理 (Logits or Probabilities -> Labels)
    if predictions.ndim > 1 and predictions.shape[-1] > 1:
        # (Batch, Classes) -> argmax -> (Batch,)
        pred_labels = np.argmax(predictions, axis=-1)
    else:
        # 既にラベル、あるいはバイナリ
        pred_labels = predictions

    # ターゲット値の形状処理 (One-hot -> Labels)
    if targets.ndim > 1 and targets.shape[-1] > 1:
        target_labels = np.argmax(targets, axis=-1)
    else:
        target_labels = targets
    
    # 形状を合わせる（必要に応じてflatten）
    pred_labels = pred_labels.flatten()
    target_labels = target_labels.flatten()

    if pred_labels.shape != target_labels.shape:
        # 形状不一致の警告（ただし計算は続行）
        logger.warning(f"Shape mismatch in accuracy calculation: pred {pred_labels.shape} vs target {target_labels.shape}")
        min_len = min(len(pred_labels), len(target_labels))
        pred_labels = pred_labels[:min_len]
        target_labels = target_labels[:min_len]

    correct = (pred_labels == target_labels).sum()
    total = len(target_labels)
    
    return float(correct / total) if total > 0 else 0.0

@MetricRegistry.register("mse")
def calculate_mse(predictions: Union[torch.Tensor, np.ndarray], 
                 targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    平均二乗誤差 (MSE)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    return float(np.mean((predictions - targets) ** 2))

@MetricRegistry.register("mae")
def calculate_mae(predictions: Union[torch.Tensor, np.ndarray], 
                 targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    平均絶対誤差 (MAE)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    return float(np.mean(np.abs(predictions - targets)))