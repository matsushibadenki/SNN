# ファイルパス: snn_research/cognitive_architecture/perception_cortex.py
# Title: Perception Cortex (Base Module)
# Description:
#   知覚野の基本クラス。
#   スパイク入力から特徴量を抽出するインターフェースおよび簡易実装を提供する。
#   多次元入力(Batch, Time, Neurons)に対応。

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
                スパイクパターン。形状は (..., num_neurons) を期待。

        Returns:
            Dict[str, torch.Tensor]:
                抽出された特徴ベクトルを含む辞書。
        """
        # 最後の次元がニューロン数と一致するか確認
        if spike_pattern.shape[-1] != self.num_neurons:
             raise ValueError(f"Input neuron count {spike_pattern.shape[-1]} mismatch with cortex {self.num_neurons}")

        # デバイス整合性の確保
        device = spike_pattern.device
        projection = self.feature_projection.to(device)

        # 1. 次元集約 (Neurons次元以外を統合して処理)
        # 入力が (Batch, Time, Neurons) などの場合、Neurons以外の次元をまとめて float化
        original_shape = spike_pattern.shape
        flat_spikes = spike_pattern.view(-1, self.num_neurons).float()

        # 2. 特徴射影 (matmul)
        # (Total_Steps, Neurons) @ (Neurons, Feature_Dim) -> (Total_Steps, Feature_Dim)
        feature_vector_flat = torch.matmul(flat_spikes, projection)

        # 3. 非線形性
        feature_vector_flat = torch.relu(feature_vector_flat)

        # 4. 元の次元（Neurons以外）に戻す
        new_shape = list(original_shape[:-1]) + [self.feature_dim]
        feature_vector = feature_vector_flat.view(*new_shape)

        return {"features": feature_vector}