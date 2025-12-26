# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: 高精度 STDP & TripletSTDP 学習則 (Shape-Aware Fix)
# 目的: 重み行列の形状(Pre,Post)に合わせて更新量の形状を自動調整し、エラーを防ぐ。

import torch
import math
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """
    標準的なペアベースSTDP（Spike-Timing-Dependent Plasticity）。
    プレとポストのスパイク時間差に基づき重みを更新する。
    """
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.a_plus = a_plus    # LTP（長期増強）強度
        self.a_minus = a_minus  # LTD（長期抑圧）強度
        self.tau_trace = tau_trace
        self.dt = dt
        
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 次元の正規化 [Batch, Neurons]
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        # トレースの初期化
        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.post_trace = torch.zeros_like(post_spikes)
        
        assert self.pre_trace is not None and self.post_trace is not None
        
        # 1. トレースの更新（時間的減衰 + 新規スパイク）
        decay = math.exp(-self.dt / self.tau_trace)
        with torch.no_grad():
            self.pre_trace = self.pre_trace * decay + pre_spikes
            self.post_trace = self.post_trace * decay + post_spikes

        # 2. 重み更新量の計算 (dw)
        # 標準的な計算: LTP = Post @ Pre.T -> (Post, Pre)
        ltp = torch.matmul(post_spikes.t(), self.pre_trace) 
        ltd = torch.matmul(self.post_trace.t(), pre_spikes)
        
        batch_size = pre_spikes.size(0)
        dw = (self.learning_rate / batch_size) * (self.a_plus * ltp - self.a_minus * ltd)
        
        # [修正] 重み行列の形状に合わせて転置を行う
        # weightsが (Pre, Post) の場合、dw (Post, Pre) を転置して (Pre, Post) にする
        if weights is not None:
            # (Pre, Post) 判定: weights[0] == pre_dim
            if weights.shape[0] == pre_spikes.shape[1] and weights.shape[1] == post_spikes.shape[1]:
                dw = dw.t()

        return dw, None

class TripletSTDP(BioLearningRule):
    """
    高精度トリプレットSTDP。
    """
    def __init__(self, learning_rate: float = 0.01, target_rate: float = 0.05, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.dt = dt
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

        # ホメオスタシス
        firing_scale = torch.clamp(cast(torch.Tensor, self.avg_firing_rate) / self.target_rate, 0.5, 2.0)
        adj_tau = self.tau_plus / firing_scale.mean()

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / adj_tau) + pre_spikes
            self.avg_firing_rate = 0.99 * cast(torch.Tensor, self.avg_firing_rate) + 0.01 * post_spikes.mean(dim=0)
        
        reward = (optional_params or {}).get("reward", 1.0)
        
        batch_size = pre_spikes.size(0)
        # dw: (Post, Pre)
        dw = (self.learning_rate * reward / batch_size) * torch.matmul(post_spikes.t(), self.pre_trace)
        
        # [修正] 重み形状に合わせた転置
        if weights is not None:
            if weights.shape[0] == pre_spikes.shape[1] and weights.shape[1] == post_spikes.shape[1]:
                dw = dw.t()
        
        return dw, None
