# ファイルパス: snn_research/core/networks/oscillatory_network.py
# 日本語タイトル: 振動ニューラルネットワーク (ONN)
# 機能説明:
#   ニューロンの「位相(Phase)」の同期を利用して計算を行う。
#   Kuramotoモデルに基づき、エネルギー最小状態への緩和を利用して
#   最適化問題（MAX-CUT等）や連想記憶を実現する。
#   BP不要、GPU不要（アナログ回路での実装に最適）。
#
#   修正: mypyエラー (Incompatible types) を解消するため、bias変数の型ヒントを追加。

import torch
import torch.nn as nn
import logging
from typing import Optional, Union  # Unionを追加

logger = logging.getLogger(__name__)


class OscillatoryNeuronGroup(nn.Module):
    """
    Kuramotoモデルに基づく振動子集団。
    dθ/dt = ω + Σ K * sin(θ_j - θ_i)
    """

    def __init__(
        self,
        num_neurons: int,
        natural_frequency: float = 1.0,
        coupling_strength: float = 0.5,
        dt: float = 0.01
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.omega = natural_frequency  # 固有角振動数
        self.K = coupling_strength     # 結合強度
        self.dt = dt

        # 位相 (0 ~ 2π)
        self.phase = nn.Parameter(torch.rand(
            num_neurons) * 2 * 3.14159, requires_grad=False)

        # 結合重み行列 (対称行列)
        self.weights = nn.Parameter(torch.randn(
            num_neurons, num_neurons), requires_grad=False)
        # 自己結合はなし
        self.weights.data.fill_diagonal_(0.0)

    def set_weights(self, weights: torch.Tensor):
        """外部から結合重みを設定（学習済みパターンや問題定義）"""
        if weights.shape != (self.num_neurons, self.num_neurons):
            raise ValueError(
                f"Weights shape mismatch. Expected ({self.num_neurons}, {self.num_neurons})")
        self.weights.data = weights
        self.weights.data.fill_diagonal_(0.0)

    def forward(self, input_phase_bias: Optional[torch.Tensor] = None, steps: int = 100) -> torch.Tensor:
        """
        時間発展を実行し、同期状態へ収束させる。
        """

        history = []

        for _ in range(steps):
            # 位相差の計算: sin(θ_j - θ_i)
            # phase.unsqueeze(0): (1, N), phase.unsqueeze(1): (N, 1) -> (N, N) broadcast
            phase_diff = self.phase.unsqueeze(0) - self.phase.unsqueeze(1)
            interaction = torch.sin(phase_diff)

            # 結合強度 * 重み * 相互作用
            # sum over j
            coupling = torch.sum(self.weights * interaction, dim=1)

            # 外部入力バイアス（特定の位相へ引き込む力など）
            # 修正: 型ヒントを明示して mypy エラーを回避
            bias: Union[float, torch.Tensor] = 0.0
            if input_phase_bias is not None:
                bias = input_phase_bias

            # Kuramoto更新式
            d_theta = self.omega + self.K * coupling + bias

            # オイラー法更新
            self.phase.data += d_theta * self.dt

            # 0-2πに正規化（必須ではないが数値安定性のため）
            self.phase.data = torch.remainder(self.phase.data, 2 * 3.14159)

            history.append(self.phase.clone())

        return torch.stack(history)

    def get_binary_state(self) -> torch.Tensor:
        """
        位相を二値状態 (-1, 1) に変換して取得する。
        π付近か0付近かで分類。
        """
        # cos(θ) > 0 -> 1, cos(θ) < 0 -> -1
        return torch.sign(torch.cos(self.phase))


class HopfieldONN:
    """
    ONNを用いたホップフィールドネットワーク（連想記憶）。
    Hebbian学習により重みを設定し、想起を行う。
    """

    def __init__(self, num_neurons: int):
        self.onn = OscillatoryNeuronGroup(num_neurons)

    def train(self, patterns: torch.Tensor):
        """
        パターンを記憶する（Hebbian Learning）。
        Patterns: (Num_Patterns, Num_Neurons), values {-1, 1}
        W_ij = (1/N) * Σ p_i * p_j
        """
        N = patterns.shape[1]
        W = torch.matmul(patterns.T, patterns) / N
        W.fill_diagonal_(0.0)
        self.onn.set_weights(W)

    def retrieve(self, noisy_pattern: torch.Tensor, steps: int = 200) -> torch.Tensor:
        """
        ノイズ混じりのパターンから記憶を想起する。
        入力パターンの値を位相の初期値としてセットし、緩和させる。
        """
        # {-1, 1} -> {π, 0} に変換して初期化
        # 1 -> 0, -1 -> π
        initial_phase = torch.where(noisy_pattern > 0, 0.0, 3.14159)

        # 少しノイズを加えて対称性を破る
        initial_phase += torch.randn_like(initial_phase) * 0.1
        self.onn.phase.data = initial_phase

        # 緩和実行
        self.onn(steps=steps)

        return self.onn.get_binary_state()
