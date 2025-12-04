# matsushibadenki/snn4/snn_research/cognitive_architecture/physics_evaluator.py
# Title: 物理法則評価器
# Description: SNNの内部状態が物理法則（滑らかさ、エネルギー効率など）に
#              どれだけ従っているかを評価し、報酬を計算するモジュール。

import torch
from typing import Dict

class PhysicsEvaluator:
    """
    SNNの内部状態の物理的一貫性を評価する。
    """
    def evaluate_physical_consistency(
        self,
        mem_sequence: torch.Tensor,
        spikes: torch.Tensor
    ) -> Dict[str, float]:
        """
        膜電位の時系列とスパイクを受け取り、物理法則に基づいた報酬を計算する。

        Args:
            mem_sequence (torch.Tensor): 膜電位の時系列データ。
            spikes (torch.Tensor): スパイク活動データ。

        Returns:
            Dict[str, float]: 各物理法則に対する報酬を格納した辞書。
        """
        # 1. 滑らかさの報酬 (Smoothness Reward)
        # 膜電位の急激な変化（時間的差分の大きさ）にペナルティを与える。
        # 変化が小さいほど報酬は1に近づく。
        if mem_sequence.numel() > 1:
            mem_diff = torch.diff(mem_sequence, dim=0)
            smoothness_penalty = torch.mean(mem_diff**2)
            smoothness_reward = torch.exp(-smoothness_penalty).item()
        else:
            smoothness_reward = 1.0

        # 2. スパース性の報酬 (Sparsity Reward / Energy Efficiency)
        # スパイク数が少ないほどエネルギー効率が良いとみなし、高い報酬を与える。
        # スパイクレートが0に近いほど報酬は1に近づく。
        spike_rate = spikes.mean()
        sparsity_reward = torch.exp(-spike_rate).item()

        return {
            "smoothness_reward": smoothness_reward,
            "sparsity_reward": sparsity_reward,
        }
