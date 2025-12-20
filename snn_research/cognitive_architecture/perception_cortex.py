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
        # print("🧠 Perception Cortex initialized.") # ログ過多を防ぐためコメントアウト推奨

    def perceive(self, spike_pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        入力されたスパイクパターンを知覚し、特徴ベクトルを抽出する。

        Args:
            spike_pattern (torch.Tensor):
                SpikeEncoderによって生成されたスパイクパターン (time_steps, num_neurons)。

        Returns:
            Dict[str, torch.Tensor]:
                抽出された特徴ベクトルを含む辞書。
                例: {'features': tensor([...])}
        """
        if spike_pattern.shape[1] != self.num_neurons:
            # 簡易チェック: 次元が合わない場合はWarningを出してリサイズなどを試みるか、エラーにする
            # ここでは厳密にチェック
             raise ValueError(f"Input neuron count {spike_pattern.shape[1]} mismatch with cortex {self.num_neurons}")

        # 1. 時間的プーリング: 時間全体のスパイク活動を集約
        temporal_features = torch.sum(spike_pattern, dim=0).float()

        # 2. 特徴射影
        feature_vector = torch.matmul(temporal_features, self.feature_projection)

        # 3. 非線形性
        feature_vector = torch.relu(feature_vector)

        return {"features": feature_vector}