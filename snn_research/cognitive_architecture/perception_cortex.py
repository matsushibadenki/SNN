# ファイルパス: snn_research/cognitive_architecture/perception_cortex.py
# Title: Perception Cortex (知覚野) モジュール
#
# Description:
# - 人工脳アーキテクチャの「知覚層」を担うコンポーネント。
# - 生のスパイクパターンをより抽象的な特徴表現に変換する。
# - 修正: nn.Moduleを継承し、PyTorchの標準的なレイヤーとして振る舞うように変更。

import torch
import torch.nn as nn
from typing import Dict

class PerceptionCortex(nn.Module): # 修正: nn.Moduleを継承
    """
    スパイクパターンから特徴を抽出する知覚野モジュール。
    """
    def __init__(self, num_neurons: int, feature_dim: int = 64):
        """
        Args:
            num_neurons (int): 入力されるスパイクパターンのニューロン数。
            feature_dim (int): 出力される特徴ベクトルの次元数。
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim
        
        # 特徴を抽出するための線形層（重み）
        # nn.Parameterとして登録することで学習可能にする
        self.feature_projection = nn.Linear(num_neurons, feature_dim)
        
        print("🧠 知覚野モジュールが初期化されました。")

    def perceive(self, spike_pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        レガシー互換のためのメソッド。内部でforwardを呼ぶ。
        """
        features = self.forward(spike_pattern)
        return {"features": features}

    def perceive_and_upload(self, spike_pattern: torch.Tensor) -> None:
        """
        互換性維持のためのメソッド。
        """
        res = self.perceive(spike_pattern)
        # アップロード処理は外部（ArtificialBrain側）で行う想定だが、
        # メソッドシグネチャ維持のため定義。
        pass

    def forward(self, spike_pattern: torch.Tensor) -> torch.Tensor:
        """
        入力されたスパイクパターンを知覚し、特徴ベクトルを抽出する。

        Args:
            spike_pattern (torch.Tensor):
                SpikeEncoderによって生成されたスパイクパターン (time_steps, batch, num_neurons) または (time_steps, num_neurons)。

        Returns:
            torch.Tensor: 抽出された特徴ベクトル。
        """
        # バッチ次元の扱い
        if spike_pattern.dim() == 2:
            # (T, N) -> (1, T, N)
            spike_pattern = spike_pattern.unsqueeze(0)
            
        # ニューロン数チェック
        if spike_pattern.shape[-1] != self.num_neurons:
             # 動的に合わせるかエラーを出す（ここではログを出してエラー）
             raise ValueError(f"入力ニューロン数不一致: Input {spike_pattern.shape[-1]} != Config {self.num_neurons}")

        # 1. 時間的プーリング: 時間軸(dim=1)でスパイクを集約 (Rate Coding)
        # (B, T, N) -> (B, N)
        temporal_features = torch.sum(spike_pattern, dim=1)

        # 2. 空間的特徴抽出
        # (B, N) -> (B, feature_dim)
        feature_vector = self.feature_projection(temporal_features)

        # 活性化関数（ReLU）
        feature_vector = torch.relu(feature_vector)

        return feature_vector
