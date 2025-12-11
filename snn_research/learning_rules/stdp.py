# ファイルパス: snn_research/learning_rules/stdp.py
# Title: STDP Learning Rule [Batch Support Fixed]
# Description:
#   バッチ入力 (Batch, Neurons) に対応したSTDPおよびTriplet STDPの実装。
#   torch.outerの代わりに行列積を使用し、バッチ全体の勾配を合計する。

import torch
from typing import Dict, Any, Optional, Tuple, cast, Union
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

    def _initialize_traces(self, pre_shape: torch.Size, post_shape: torch.Size, device: torch.device):
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
        
        # 1D入力の場合はバッチ次元(1)を追加して統一的に扱う
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)

        # トレース初期化
        if (self.pre_trace is None or self.post_trace is None or 
            self.pre_trace.shape != pre_spikes.shape or self.post_trace.shape != post_spikes.shape):
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        # トレース更新
        self._update_traces(pre_spikes, post_spikes)

        assert self.pre_trace is not None and self.post_trace is not None

        # 重み更新量の計算 (Batch dimension is summed over)
        # Weights: (Post, Pre)
        # Pre: (Batch, Pre), Post: (Batch, Post)
        
        # LTP: Post spikes * Pre trace
        # (Batch, Post).T @ (Batch, Pre) -> (Post, Batch) @ (Batch, Pre) -> (Post, Pre)
        ltp = torch.matmul(post_spikes.t(), self.pre_trace)
        
        # LTD: Post trace * Pre spikes
        # (Batch, Post).T @ (Batch, Pre) -> (Post, Pre)
        ltd = torch.matmul(self.post_trace.t(), pre_spikes)

        dw = self.learning_rate * (self.a_plus * ltp - self.a_minus * ltd)
        
        return dw, None


class TripletSTDP(BioLearningRule):
    """
    Triplet STDP (トリプレットSTDP) 学習則 with Homeostasis.
    Batch対応版。
    """
    pre_trace: Optional[torch.Tensor]
    post_trace: Optional[torch.Tensor]
    pre_trace_triplet: Optional[torch.Tensor]
    post_trace_triplet: Optional[torch.Tensor]
    avg_firing_rate: Optional[torch.Tensor]

    def __init__(
        self, 
        learning_rate: float, 
        a_plus_pair: float, 
        a_minus_pair: float, 
        tau_trace_pair: float,
        a_plus_triplet: float,
        a_minus_triplet: float,
        tau_trace_triplet: float,
        dt: float = 1.0,
        target_rate: float = 0.05, 
        homeostasis_strength: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.dt = dt
        
        self.a_plus_pair = a_plus_pair
        self.a_minus_pair = a_minus_pair
        self.tau_trace_pair = tau_trace_pair
        
        self.a_plus_triplet = a_plus_triplet   
        self.a_minus_triplet = a_minus_triplet 
        self.tau_trace_triplet = tau_trace_triplet 
        
        self.target_rate = target_rate
        self.homeostasis_strength = homeostasis_strength

        self.pre_trace = None
        self.post_trace = None
        self.pre_trace_triplet = None
        self.post_trace_triplet = None
        self.avg_firing_rate = None

    def _initialize_traces(self, pre_shape: torch.Size, post_shape: torch.Size, device: torch.device):
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        self.pre_trace_triplet = torch.zeros(pre_shape, device=device)
        self.post_trace_triplet = torch.zeros(post_shape, device=device)
        
        # Firing rate is averaged over batch? Or keep per neuron? 
        # Usually homeostasis is per neuron. 
        # avg_firing_rate shape: (Post,) derived from (Batch, Post).
        self.avg_firing_rate = torch.full((post_shape[1],), self.target_rate, device=device)
        
    def _update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        pre_trace = cast(torch.Tensor, self.pre_trace)
        post_trace = cast(torch.Tensor, self.post_trace)
        pre_trace_triplet = cast(torch.Tensor, self.pre_trace_triplet)
        post_trace_triplet = cast(torch.Tensor, self.post_trace_triplet)
        avg_firing_rate = cast(torch.Tensor, self.avg_firing_rate)

        pre_trace = pre_trace - (pre_trace / self.tau_trace_pair) * self.dt + pre_spikes
        post_trace = post_trace - (post_trace / self.tau_trace_pair) * self.dt + post_spikes
        
        pre_trace_triplet = pre_trace_triplet - (pre_trace_triplet / self.tau_trace_triplet) * self.dt + pre_spikes
        post_trace_triplet = post_trace_triplet - (post_trace_triplet / self.tau_trace_triplet) * self.dt + post_spikes
        
        # Average firing rate update (Batch mean -> Scalar per neuron)
        batch_mean_spike = post_spikes.mean(dim=0) # (Batch, Post) -> (Post,)
        alpha = 0.001 
        avg_firing_rate = (1 - alpha) * avg_firing_rate + alpha * batch_mean_spike
        
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
        
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)
        
        if (self.pre_trace is None or 
            self.pre_trace.shape != pre_spikes.shape):
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        self._update_traces(pre_spikes, post_spikes)

        pre_trace = cast(torch.Tensor, self.pre_trace)
        post_trace = cast(torch.Tensor, self.post_trace)
        pre_trace_triplet = cast(torch.Tensor, self.pre_trace_triplet)
        post_trace_triplet = cast(torch.Tensor, self.post_trace_triplet)
        avg_firing_rate = cast(torch.Tensor, self.avg_firing_rate)

        # Batch Matmul for updates
        # 1. Pairwise
        # LTP: Post * Pre_trace
        dw_pair_ltp = torch.matmul(post_spikes.t(), pre_trace)
        # LTD: Post_trace * Pre
        dw_pair_ltd = torch.matmul(post_trace.t(), pre_spikes)
        
        # 2. Triplet
        # LTP (pre-post-pre): Pre * Post_trace_triplet -> (Pre, Post).T -> (Post, Pre)
        # (Post_trace_triplet.T @ Pre)
        dw_triplet_ltd = torch.matmul(post_trace_triplet.t(), pre_spikes)
        
        # LTD (post-pre-post): Post * Pre_trace_triplet
        dw_triplet_ltp = torch.matmul(post_spikes.t(), pre_trace_triplet)

        dw = (self.a_plus_pair * dw_pair_ltp - self.a_minus_pair * dw_pair_ltd)
        dw += (-self.a_minus_triplet * dw_triplet_ltd + self.a_plus_triplet * dw_triplet_ltp)

        # 3. Homeostasis
        rate_factor = (avg_firing_rate - self.target_rate) * self.homeostasis_strength
        homeostasis_mod = 1.0 - rate_factor.unsqueeze(1)
        
        dw = self.learning_rate * dw * homeostasis_mod
        
        return dw, None
