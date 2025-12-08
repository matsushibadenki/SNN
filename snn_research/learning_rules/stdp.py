# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: STDPおよびTriplet STDP学習則 (恒常性維持付き)
# 機能説明: 
#   標準STDPおよびTriplet STDPの実装。
#   Triplet STDPには、発火率に応じた恒常性維持 (Homeostasis) 項を追加し、
#   ネットワークの安定性を向上。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """ペアベースのSTDP学習ルールを実装するクラス。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_trace = tau_trace
        self.dt = dt
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def _initialize_traces(self, pre_shape: int, post_shape: int, device: torch.device):
        """スパイクトレースを初期化する。"""
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        
    def _update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """スパイクトレースを更新する。"""
        assert self.pre_trace is not None and self.post_trace is not None, "Traces not initialized."
            
        self.pre_trace = self.pre_trace - (self.pre_trace / self.tau_trace) * self.dt + pre_spikes
        self.post_trace = self.post_trace - (self.post_trace / self.tau_trace) * self.dt + post_spikes

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STDPに基づいて重み変化量を計算する。"""
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape[0] != pre_spikes.shape[0] or self.post_trace.shape[0] != post_spikes.shape[0]:
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)

        self._update_traces(pre_spikes, post_spikes)

        assert self.pre_trace is not None and self.post_trace is not None

        dw = torch.zeros_like(weights)
        dw += self.learning_rate * self.a_plus * torch.outer(post_spikes, self.pre_trace)
        dw -= self.learning_rate * self.a_minus * torch.outer(pre_spikes, self.post_trace).T
        
        return dw, None


class TripletSTDP(BioLearningRule):
    """
    Triplet STDP (トリプレットSTDP) 学習則 with Homeostasis.
    doc/プロジェクト強化案の調査.md (セクション2.1, 引用[18, 22]) に基づく。
    標準的なペアワイズSTDPに加え、3つのスパイクの相互作用（トリプレット項）を考慮し、
    さらに発火率に基づいた恒常性維持を行う。
    """
    pre_trace: Optional[torch.Tensor]
    post_trace: Optional[torch.Tensor]
    pre_trace_triplet: Optional[torch.Tensor]
    post_trace_triplet: Optional[torch.Tensor]
    
    # Homeostasis用
    avg_firing_rate: Optional[torch.Tensor]

    def __init__(
        self, 
        learning_rate: float, 
        # ペアワイズ項
        a_plus_pair: float, 
        a_minus_pair: float, 
        tau_trace_pair: float,
        # トリプレット項
        a_plus_triplet: float,
        a_minus_triplet: float,
        tau_trace_triplet: float,
        dt: float = 1.0,
        # Homeostasis項
        target_rate: float = 0.05, # 目標発火率
        homeostasis_strength: float = 0.1 # 調整強度
    ):
        self.learning_rate = learning_rate
        self.dt = dt
        
        # ペアワイズ項のパラメータ
        self.a_plus_pair = a_plus_pair
        self.a_minus_pair = a_minus_pair
        self.tau_trace_pair = tau_trace_pair
        
        # トリプレット項のパラメータ
        self.a_plus_triplet = a_plus_triplet   # y (post-pre-post)
        self.a_minus_triplet = a_minus_triplet # x (pre-post-pre)
        self.tau_trace_triplet = tau_trace_triplet 
        
        # Homeostasis
        self.target_rate = target_rate
        self.homeostasis_strength = homeostasis_strength

        self.pre_trace = None
        self.post_trace = None
        self.pre_trace_triplet = None
        self.post_trace_triplet = None
        self.avg_firing_rate = None

    def _initialize_traces(self, pre_shape: int, post_shape: int, device: torch.device):
        """スパイクトレースと平均発火率を初期化する。"""
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        self.pre_trace_triplet = torch.zeros(pre_shape, device=device)
        self.post_trace_triplet = torch.zeros(post_shape, device=device)
        self.avg_firing_rate = torch.full((post_shape,), self.target_rate, device=device)
        
    def _update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """スパイクトレースを更新する。"""
        # (型チェック)
        pre_trace = cast(torch.Tensor, self.pre_trace)
        post_trace = cast(torch.Tensor, self.post_trace)
        pre_trace_triplet = cast(torch.Tensor, self.pre_trace_triplet)
        post_trace_triplet = cast(torch.Tensor, self.post_trace_triplet)
        avg_firing_rate = cast(torch.Tensor, self.avg_firing_rate)

        # ペアワイズ トレースの更新
        pre_trace = pre_trace - (pre_trace / self.tau_trace_pair) * self.dt + pre_spikes
        post_trace = post_trace - (post_trace / self.tau_trace_pair) * self.dt + post_spikes
        
        # トリプレット トレースの更新
        pre_trace_triplet = pre_trace_triplet - (pre_trace_triplet / self.tau_trace_triplet) * self.dt + pre_spikes
        post_trace_triplet = post_trace_triplet - (post_trace_triplet / self.tau_trace_triplet) * self.dt + post_spikes
        
        # 平均発火率の更新 (移動平均)
        # alpha = dt / tau_avg (tau_avg ~ 1000 steps)
        alpha = 0.001 
        avg_firing_rate = (1 - alpha) * avg_firing_rate + alpha * post_spikes
        
        # 属性に再代入
        self.pre_trace = pre_trace
        self.post_trace = post_trace
        self.pre_trace_triplet = pre_trace_triplet
        self.post_trace_triplet = post_trace_triplet
        self.avg_firing_rate = avg_firing_rate


    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Triplet STDP + Homeostasis に基づいて重み変化量を計算する。"""
        
        if (self.pre_trace is None or self.post_trace is None or 
            self.pre_trace_triplet is None or self.post_trace_triplet is None or
            self.pre_trace.shape[0] != pre_spikes.shape[0] or 
            self.post_trace.shape[0] != post_spikes.shape[0]):
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)

        # トレースを更新
        self._update_traces(pre_spikes, post_spikes)

        # (型チェック)
        pre_trace = cast(torch.Tensor, self.pre_trace)
        post_trace = cast(torch.Tensor, self.post_trace)
        pre_trace_triplet = cast(torch.Tensor, self.pre_trace_triplet)
        post_trace_triplet = cast(torch.Tensor, self.post_trace_triplet)
        avg_firing_rate = cast(torch.Tensor, self.avg_firing_rate)

        dw = torch.zeros_like(weights)
        
        # --- 1. ペアワイズ項 (標準STDP) ---
        # LTP (Post-then-Pre): post_spikes * pre_trace
        dw += self.a_plus_pair * torch.outer(post_spikes, pre_trace)
        # LTD (Pre-then-Post): pre_spikes * post_trace
        dw -= self.a_minus_pair * torch.outer(pre_spikes, post_trace).T
        
        # --- 2. トリプレット項 (引用[22]に基づく) ---
        # LTP (pre-post-pre): pre_spikes * post_trace_triplet
        dw -= self.a_minus_triplet * torch.outer(pre_spikes, post_trace_triplet).T
        # LTD (post-pre-post): post_spikes * pre_trace_triplet
        dw += self.a_plus_triplet * torch.outer(post_spikes, pre_trace_triplet)

        # --- 3. 恒常性維持 (Homeostasis) ---
        # 発火率が目標より高い -> LTDを強化 (dwを減らす)
        # 発火率が目標より低い -> LTPを強化 (dwを増やす)
        # rate_factor: > 0 なら過活動(抑制必要), < 0 なら低活動(興奮必要)
        rate_factor = (avg_firing_rate - self.target_rate) * self.homeostasis_strength
        
        # 全シナプスに対して調整 (post依存)
        # dw -= rate_factor.unsqueeze(1) * abs(dw) # 変化量に比例させるか、定数か
        # シンプルに学習率を変調する形
        homeostasis_mod = 1.0 - rate_factor.unsqueeze(1)
        
        # 学習率を適用
        dw = self.learning_rate * dw * homeostasis_mod
        
        return dw, None
