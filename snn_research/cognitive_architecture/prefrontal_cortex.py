# ファイルパス: snn_research/cognitive_architecture/perception_cortex.py
# Title: Perception Cortex (Base Module)
# Description:
#   知覚野の基本クラス。
#   スパイク入力から特徴量を抽出するインターフェースおよび簡易実装を提供する。

import torch
from typing import Dict

class PerceptionCortex:
    """
    スパイクパターンから特徴を抽出する知覚野モジュール。
    """
    def __init__(self, num_neurons: int, feature_dim: int = 64):
        """
        Args:
            num_neurons (int): 入力されるスパイクパターンのニューロン数。
            feature_dim (int): 出力される特徴ベクトルの次元数。
        """
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim
        # 特徴を抽出するための簡易的な線形層（重み）
        self.feature_projection = torch.randn((num_neurons, feature_dim))

    def perceive(self, spike_pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        入力されたスパイクパターンを知覚し、特徴ベクトルを抽出する。

        Args:
            spike_pattern (torch.Tensor):
                スパイクパターン。期待される形状: (Batch, num_neurons) 
                または (Time, num_neurons)、あるいは (Batch, Time, num_neurons)

        Returns:
            Dict[str, torch.Tensor]:
                抽出された特徴ベクトルを含む辞書。
        """
        # デバイスの整合性を確保
        projection = self.feature_projection.to(spike_pattern.device)

        # 入力次元の整理
        # matmul (A, B) で Bが (784, 256) の場合、Aの最後次元が 784 である必要がある
        if spike_pattern.shape[-1] != self.num_neurons:
             raise ValueError(f"Input neuron count {spike_pattern.shape[-1]} mismatch with cortex {self.num_neurons}")

        # 1. 時間的/空間的プーリング
        # 3次元 (Batch, Time, Neurons) の場合は Time(dim=1) で集約
        # 2次元 (Batch, Neurons) の場合はそのまま
        if spike_pattern.ndim == 3:
            aggregated_features = torch.sum(spike_pattern, dim=1).float()
        elif spike_pattern.ndim == 2:
            aggregated_features = spike_pattern.float()
        else:
            aggregated_features = spike_pattern.view(-1, self.num_neurons).float()

        # 2. 特徴射影 (matmul)
        # aggregated_features: (Batch, num_neurons)
        # projection: (num_neurons, feature_dim)
        feature_vector = torch.matmul(aggregated_features, projection)

        # 3. 非線形性
        feature_vector = torch.relu(feature_vector)

        return {"features": feature_vector}
