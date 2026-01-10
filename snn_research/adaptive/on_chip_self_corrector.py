# ファイルパス: snn_research/adaptive/on_chip_self_corrector.py
# Title: On-Chip Self Corrector (LNN/RSNN Engine)
# Description:
# - Objective 5 & 6: 非勾配型学習システム(Non-gradient Learning)。
# - バックプロパゲーションを使わず、局所的なスパイク活動と報酬信号のみで重みを更新する。
# - Liquid State Machine / Reservoir Computing の概念を拡張した自己組織化ロジック。

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class OnChipSelfCorrector(nn.Module):
    """
    オンチップ自己修正モジュール。
    推論中にリアルタイムでシナプス荷重を微調整し、環境に適応する。
    """

    def __init__(self, learning_rate: float = 1e-4, stdp_window: int = 20, device: str = 'cpu'):
        super().__init__()
        self.lr = learning_rate
        self.window = stdp_window
        self.device = device

        # 報酬予測誤差 (RPE) の履歴
        self.rpe_trace = 0.0

        logger.info("🔧 On-Chip Self Corrector initialized (Non-gradient mode).")

    def observe_and_correct(self,
                            layer_weights: torch.Tensor,
                            pre_spikes: torch.Tensor,
                            post_spikes: torch.Tensor,
                            reward_signal: float) -> torch.Tensor:
        """
        スパイク活動と報酬に基づいて重みを修正する (R-STDP: Reward-modulated STDP)。

        Args:
            layer_weights: 対象レイヤーの重み (参照渡し想定だが、ここでは更新後のTensorを返す)
            pre_spikes: プレニューロンのスパイク履歴 [Batch, Time, In_Features]
            post_spikes: ポストニューロンのスパイク履歴 [Batch, Time, Out_Features]
            reward_signal: 環境からの報酬 (-1.0 to 1.0)

        Returns:
            updated_weights: 更新された重み
        """
        # 勾配計算を無効化 (完全な推論モードでの学習)
        with torch.no_grad():
            # 簡易的な同時発火検出 (Correlation-based Hebbian term)
            # Pre[b, t, i] * Post[b, t, j] -> Weight[Out, In] の相関
            # Note: PyTorch Linear weight is [Out_features, In_features]

            # 時間次元での平均活動率
            pre_rate = pre_spikes.float().mean(dim=(0, 1))  # [In]
            post_rate = post_spikes.float().mean(dim=(0, 1))  # [Out]

            # ヘブ則項: "Fire together, wire together"
            # outer(post, pre) で [Out, In] の形状を作成
            hebbian_term = torch.outer(post_rate, pre_rate)  # [Out, In]

            # 恒常性維持項 (LTD): 発火しすぎを防ぐ
            # Post側の発火率が高い場合、全体的に抑制する
            homeostatic_term = post_rate.unsqueeze(
                1) * 0.1  # [Out, 1] -> broadcast

            # ドーパミン変調 (Reward Modulation)
            # 報酬が正なら強化、負なら抑制 (Anti-Hebbian)
            modulation = reward_signal

            # 3要素則 (Three-factor rule) の適用
            # delta_w = LearningRate * Reward * (Hebbian - Homeostatic)
            delta_w = self.lr * modulation * (hebbian_term - homeostatic_term)

            # 重みの更新
            new_weights = layer_weights + delta_w.to(layer_weights.device)

            # 重みのクリッピング (発散防止)
            new_weights = torch.clamp(new_weights, -1.0, 1.0)

            return new_weights

    def compute_local_error(self, desired_activity: torch.Tensor, actual_activity: torch.Tensor) -> float:
        """
        局所的な予測誤差を計算する (Predictive Coding的アプローチ)。
        """
        with torch.no_grad():
            error = torch.mean(
                (desired_activity - actual_activity) ** 2).item()
        return error
