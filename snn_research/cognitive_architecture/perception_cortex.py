# ファイルパス: snn_research/cognitive_architecture/perception_cortex.py
# (新規作成)
#
# Title: Perception Cortex (知覚野) モジュール
#
# Description:
# - 人工脳アーキテクチャの「知覚層」を担うコンポーネント。
# - 符号化層から受け取った生のスパイクパターンを処理し、
#   より抽象的な「特徴表現」に変換する。
# - この実装では、将来的に複雑なSNN+CNNハイブリッドモデルに置き換えることを見据え、
#   スパイクの時間的・空間的なプーリングを行うことで特徴抽出を簡易的にシミュレートする。

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
        print("🧠 知覚野モジュールが初期化されました。")

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
            raise ValueError(f"入力スパイクのニューロン数 ({spike_pattern.shape[1]}) が"
                             f"知覚野のニューロン数 ({self.num_neurons}) と一致しません。")

        print("👀 知覚野: スパイクパターンから特徴を抽出しています...")

        # 1. 時間的プーリング: 時間全体のスパイク活動を集約
        #    各ニューロンの発火総数を計算
        temporal_features = torch.sum(spike_pattern, dim=0)

        # 2. 空間的プーリング（特徴射影）:
        #    集約された活動を、より低次元の特徴空間に射影する
        #    (簡易的な全結合層の役割)
        feature_vector = torch.matmul(temporal_features, self.feature_projection)

        # 活性化関数（例: ReLU）を適用して非線形性を導入
        feature_vector = torch.relu(feature_vector)

        print(f"  - {self.feature_dim}次元の特徴ベクトルを生成しました。")

        return {"features": feature_vector}