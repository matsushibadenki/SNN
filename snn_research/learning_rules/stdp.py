# snn_research/learning_rules/stdp.py
# (修正)
# 修正: 階層的因果学習のため、戻り値の型を基底クラスに合わせる。
#
# 改善 (v2):
# - doc/プロジェクト強化案の調査.md (セクション2.1, 引用[18, 22]) に基づき、
#   TripletSTDP (トリプレットSTDP) 学習則を実装。 (ロードマップ未実装機能)

import torch
# --- ▼ 修正 (v2) ▼ ---
from typing import Dict, Any, Optional, Tuple, cast
# --- ▲ 修正 (v2) ▲ ---
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """ペアベースのSTDP学習ルールを実装するクラス。"""
    # ... (init, _initialize_traces, _update_traces は変更なし) ...
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

# --- ▼▼▼ 改善 (v2): TripletSTDP の実装 ▼▼▼ ---
class TripletSTDP(BioLearningRule):
    """
    Triplet STDP (トリプレットSTDP) 学習則。
    doc/プロジェクト強化案の調査.md (セクション2.1, 引用[18, 22]) に基づく。
    標準的なペアワイズSTDPに加え、3つのスパイクの相互作用（トリプレット項）を考慮する。
    """
    pre_trace: Optional[torch.Tensor]
    post_trace: Optional[torch.Tensor]
    pre_trace_triplet: Optional[torch.Tensor]
    post_trace_triplet: Optional[torch.Tensor]

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
        dt: float = 1.0
    ):
        self.learning_rate = learning_rate
        self.dt = dt
        
        # ペアワイズ項のパラメータ
        self.a_plus_pair = a_plus_pair
        self.a_minus_pair = a_minus_pair
        self.tau_trace_pair = tau_trace_pair
        
        # トリプレット項のパラメータ (引用[22]に基づく)
        self.a_plus_triplet = a_plus_triplet   # y (post-pre-post)
        self.a_minus_triplet = a_minus_triplet # x (pre-post-pre)
        self.tau_trace_triplet = tau_trace_triplet # tau_x, tau_y

        self.pre_trace = None
        self.post_trace = None
        self.pre_trace_triplet = None
        self.post_trace_triplet = None

    def _initialize_traces(self, pre_shape: int, post_shape: int, device: torch.device):
        """ペアワイズおよびトリプレット用のスパイクトレースを初期化する。"""
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        self.pre_trace_triplet = torch.zeros(pre_shape, device=device)
        self.post_trace_triplet = torch.zeros(post_shape, device=device)
        
    def _update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """スパイクトレースを更新する。"""
        # (型チェック)
        pre_trace = cast(torch.Tensor, self.pre_trace)
        post_trace = cast(torch.Tensor, self.post_trace)
        pre_trace_triplet = cast(torch.Tensor, self.pre_trace_triplet)
        post_trace_triplet = cast(torch.Tensor, self.post_trace_triplet)

        # ペアワイズ トレースの更新
        pre_trace = pre_trace - (pre_trace / self.tau_trace_pair) * self.dt + pre_spikes
        post_trace = post_trace - (post_trace / self.tau_trace_pair) * self.dt + post_spikes
        
        # トリプレット トレースの更新
        pre_trace_triplet = pre_trace_triplet - (pre_trace_triplet / self.tau_trace_triplet) * self.dt + pre_spikes
        post_trace_triplet = post_trace_triplet - (post_trace_triplet / self.tau_trace_triplet) * self.dt + post_spikes
        
        # 属性に再代入
        self.pre_trace = pre_trace
        self.post_trace = post_trace
        self.pre_trace_triplet = pre_trace_triplet
        self.post_trace_triplet = post_trace_triplet


    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Triplet STDPに基づいて重み変化量を計算する。"""
        
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

        # 学習率を適用
        dw = self.learning_rate * dw
        
        return dw, None
# --- ▲▲▲ 改善 (v2) ▲▲▲ ---