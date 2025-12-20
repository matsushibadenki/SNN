# ファイルパス: snn_research/core/synapse_dynamics.py
# (修正: apply_probabilistic_transmission に training 引数を追加)
#
# Title: シナプスダイナミクス (可塑性)
# Description:
# - SNNの学習において重要な、シナプスの動的な性質をモデル化する。
# - STDP, STP, および確率的伝達の実装。
#
# mypy --strict 準拠。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

def apply_probabilistic_transmission(x: torch.Tensor, reliability: float = 1.0, training: bool = True) -> torch.Tensor:
    """
    確率的なシナプス伝達をシミュレートする。
    入力信号 x に対して、確率 reliability で信号を通過させるドロップアウトマスクを適用する。
    
    Args:
        x (torch.Tensor): 入力スパイクまたは電流。
        reliability (float): 伝達成功確率 (0.0 ~ 1.0)。
                             1.0 なら全ての信号が通過 (変化なし)。
                             0.0 なら全ての信号が遮断 (ゼロ)。
        training (bool): 学習モードかどうか。
                         生物学的シミュレーションとしては常に適用する場合もあるが、
                         互換性のため引数として受け取る (デフォルトは True)。
    
    Returns:
        torch.Tensor: 確率的にマスクされた出力。
    """
    if reliability >= 1.0:
        return x
    if reliability <= 0.0:
        return torch.zeros_like(x)
    
    # trainingフラグを考慮するかどうかは設計次第だが、
    # ここでは生物学的妥当性を重視し、training=Falseでも確率的挙動を維持する実装とする。
    # (もし推論時に決定論的にしたい場合は `if not training: return x * reliability` とする)
    
    if x.is_floating_point():
        mask = (torch.rand_like(x) < reliability).float()
    else:
        # 整数型やブール型の場合
        mask = (torch.rand_like(x.float()) < reliability).to(x.dtype)
        
    return x * mask


class SynapseDynamics(nn.Module):
    """
    シナプスダイナミクスの基底クラス。
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weight: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """
        重みの更新、または動的な重みの適用を行う。
        """
        raise NotImplementedError


class STDP(SynapseDynamics):
    """
    Spike-Timing Dependent Plasticity (STDP) 実装。
    トレースベースの簡易実装。
    """
    pre_trace: torch.Tensor
    post_trace: torch.Tensor

    def __init__(
        self, 
        learning_rate: float = 1e-4,
        tau_pre: float = 20.0, 
        tau_post: float = 20.0,
        w_max: float = 1.0,
        w_min: float = -1.0
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        self.w_min = w_min
        
        self.register_buffer('pre_trace', torch.zeros(1))
        self.register_buffer('post_trace', torch.zeros(1))

    def reset_state(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def update(
        self, 
        weight: torch.nn.Parameter, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor
    ) -> None:
        """
        STDP則に基づいて重みを更新する。
        pre_spikes: (Batch, In_Features)
        post_spikes: (Batch, Out_Features)
        weight: (Out_Features, In_Features)
        """
        batch_size = pre_spikes.shape[0]
        
        if self.pre_trace.shape[0] != batch_size or self.pre_trace.shape[1] != pre_spikes.shape[1]:
            self.pre_trace = torch.zeros_like(pre_spikes)
        if self.post_trace.shape[0] != batch_size or self.post_trace.shape[1] != post_spikes.shape[1]:
            self.post_trace = torch.zeros_like(post_spikes)

        self.pre_trace = self.pre_trace * (1.0 - 1.0/self.tau_pre) + pre_spikes
        self.post_trace = self.post_trace * (1.0 - 1.0/self.tau_post) + post_spikes

        delta_w_ltp = torch.bmm(post_spikes.unsqueeze(2), self.pre_trace.unsqueeze(1)).mean(dim=0)
        delta_w_ltd = torch.bmm(self.post_trace.unsqueeze(2), pre_spikes.unsqueeze(1)).mean(dim=0)

        delta_w = self.learning_rate * (delta_w_ltp - delta_w_ltd)

        with torch.no_grad():
            weight.add_(delta_w)
            weight.clamp_(self.w_min, self.w_max)


class STP(SynapseDynamics):
    """
    Short-Term Plasticity (STP) 実装。
    Markram et al. (1998) モデルに基づく。
    """
    u: torch.Tensor
    x: torch.Tensor

    def __init__(self, features: int, tau_f: float = 50.0, tau_d: float = 200.0, U: float = 0.2):
        super().__init__()
        self.features = features
        self.tau_f = tau_f 
        self.tau_d = tau_d 
        self.U = U         
        
        self.register_buffer('u', torch.zeros(1))
        self.register_buffer('x', torch.zeros(1))

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.u = torch.ones(batch_size, self.features, device=device) * self.U
        self.x = torch.ones(batch_size, self.features, device=device)

    def forward(self, weight: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.u.ndim == 0 or self.u.shape[0] != pre_spikes.shape[0]:
            self.reset_state(pre_spikes.shape[0], pre_spikes.device)

        decay_f = torch.exp(torch.tensor(-1.0 / self.tau_f, device=weight.device))
        decay_d = torch.exp(torch.tensor(-1.0 / self.tau_d, device=weight.device))
        
        self.u = self.U + (self.u - self.U) * decay_f
        self.x = 1.0 + (self.x - 1.0) * decay_d
        
        gain = self.u * self.x
        
        spike_mask = pre_spikes > 0
        
        u_plus = self.u + self.U * (1.0 - self.u)
        x_minus = self.x - self.u * self.x 
        
        self.u = torch.where(spike_mask, u_plus, self.u)
        self.x = torch.where(spike_mask, x_minus, self.x)
        
        modulated_spikes = pre_spikes * gain
        effective_input = torch.matmul(modulated_spikes, weight.t())
        
        return effective_input