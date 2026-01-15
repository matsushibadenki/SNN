# snn_research/learning_rules/reward_modulated_stdp.py
# 修正: RewardModulatedSTDP クラスを復元し、既存コードとの互換性を維持。
#       同時に EmotionModulatedSTDP クラスも定義。

import torch

class RewardModulatedSTDP:
    """
    従来の報酬変調型STDP。
    既存コードの互換性のために維持。
    """
    def __init__(self, learning_rate: float = 1e-4, time_window: int = 20):
        self.lr = learning_rate
        self.window = time_window
        self.eligibility_trace = None

    def update(self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0):
        """
        標準的な R-STDP 更新。
        """
        # STDP項の計算 (簡易版: post * pre)
        # pre_spikes: (Batch, In)
        # post_spikes: (Batch, Out)
        # stdp: (Out, In)
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        # トレースの減衰と更新
        self.eligibility_trace = self.eligibility_trace * 0.9 + stdp_update
        
        # 報酬によるゲート
        delta_w = self.lr * self.eligibility_trace * reward
        
        return weights + delta_w


class EmotionModulatedSTDP(RewardModulatedSTDP):
    """
    イリヤ・サツケバーの仮説に基づく感情変調型STDP。
    外部報酬だけでなく、内部価値（感情）によっても学習が進む。
    """
    def update(self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
               external_reward: float = 0.0, internal_value: float = 0.0):
        """
        Args:
            external_reward: 環境からの報酬
            internal_value: 扁桃体等からの内部価値信号
        """
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        self.eligibility_trace = self.eligibility_trace * 0.9 + stdp_update
        
        # 報酬シグナル = 外部報酬 + α * 内部価値
        # 外部報酬がゼロでも、内部価値があれば学習が駆動される
        alpha = 1.0 
        modulation_signal = external_reward + alpha * internal_value
        
        delta_w = self.lr * self.eligibility_trace * modulation_signal
        
        return weights + delta_w