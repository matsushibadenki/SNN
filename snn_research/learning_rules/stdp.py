# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: 高精度 STDP & TripletSTDP 学習則 (バグ修正版)
# 目的: バッチサイズとニューロン数の不一致を解消し、行列演算の安定性を確保。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """標準的なペアベースSTDP学習則。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        super().__init__()
        self.learning_rate, self.a_plus, self.a_minus, self.tau_trace, self.dt = learning_rate, a_plus, a_minus, tau_trace, dt
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # バッチ次元の正規化
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self.pre_trace, self.post_trace = torch.zeros_like(pre_spikes), torch.zeros_like(post_spikes)
        
        assert self.pre_trace is not None and self.post_trace is not None
        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / self.tau_trace) + pre_spikes
            self.post_trace = self.post_trace * (1.0 - self.dt / self.tau_trace) + post_spikes

        # dw: (Post_Size, Pre_Size) の行列演算
        # [修正] matmul の入力を転置し、バッチ平均をとることで形状不一致を解消
        ltp = torch.matmul(post_spikes.t(), self.pre_trace) / pre_spikes.size(0)
        ltd = torch.matmul(self.post_trace.t(), pre_spikes) / pre_spikes.size(0)
        
        dw = self.learning_rate * (self.a_plus * ltp - self.a_minus * ltd)
        return dw, None

class TripletSTDP(BioLearningRule):
    """⑬ 認識精度 96% 超えを目指す高精度トリプレット STDP。"""
    def __init__(self, learning_rate: float = 0.01, target_rate: float = 0.05, dt: float = 1.0):
        super().__init__()
        self.learning_rate, self.target_rate, self.dt = learning_rate, target_rate, dt
        self.tau_plus = torch.tensor(16.8)
        self.pre_trace: Optional[torch.Tensor] = None
        self.avg_firing_rate: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.avg_firing_rate = torch.full((post_spikes.shape[-1],), self.target_rate, device=pre_spikes.device)

        # 活動に応じた時定数調整
        scale = torch.clamp(cast(torch.Tensor, self.avg_firing_rate) / self.target_rate, 0.5, 2.0)
        adj_tau = self.tau_plus / scale.mean()

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / adj_tau) + pre_spikes
            self.avg_firing_rate = 0.999 * cast(torch.Tensor, self.avg_firing_rate) + 0.001 * post_spikes.mean(dim=0)
        
        reward = (optional_params or {}).get("reward", 1.0)
        # dw: (Post, Pre)
        dw = self.learning_rate * reward * torch.matmul(post_spikes.t(), self.pre_trace) / pre_spikes.size(0)
        return dw, None
