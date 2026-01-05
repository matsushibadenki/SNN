# ファイルパス: snn_research/core/synapse_dynamics.py
# Title: シナプスダイナミクス (可塑性 & ホメオスタシス)
# Description:
# - STDP, STP に加え、Homeostatic Plasticity (Synaptic Scaling) を実装。
# - 生体の恒常性維持機構を模倣し、学習の安定性を劇的に向上させる。

import torch
import torch.nn as nn
from typing import Optional

def apply_probabilistic_transmission(x: torch.Tensor, reliability: float = 1.0, training: bool = True) -> torch.Tensor:
    """確率的なシナプス伝達をシミュレートする"""
    if reliability >= 1.0:
        return x
    if reliability <= 0.0:
        return torch.zeros_like(x)
    
    if x.is_floating_point():
        mask = (torch.rand_like(x) < reliability).float()
    else:
        mask = (torch.rand_like(x.float()) < reliability).to(x.dtype)
        
    return x * mask


class SynapseDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weight: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class HomeostaticPlasticity(SynapseDynamics):
    """
    Synaptic Scaling (Homeostatic Plasticity).
    ニューロンの長期的な発火率を監視し、目標発火率に近づけるように
    シナプス重み全体を乗法的にスケーリングする。
    
    Delta W = alpha * (Target_Rate - Avg_Rate) * W
    """
    avg_firing_rate: torch.Tensor

    def __init__(self, features: int, target_rate: float = 0.05, alpha: float = 0.01, decay: float = 0.99):
        super().__init__()
        self.features = features
        self.target_rate = target_rate
        self.alpha = alpha
        self.decay = decay
        
        # 移動平均発火率の保存
        self.register_buffer('avg_firing_rate', torch.zeros(features))

    def update(self, weight: torch.nn.Parameter, post_spikes: torch.Tensor) -> None:
        """
        Args:
            weight: (Out, In)
            post_spikes: (Batch, Out)
        """
        # 現在のバッチでの発火率 (Batch方向の平均)
        current_rate = post_spikes.float().mean(dim=0)
        
        # 移動平均の更新
        self.avg_firing_rate = self.avg_firing_rate * self.decay + current_rate * (1.0 - self.decay)
        
        # スケーリング係数の計算
        # 発火率が低すぎる -> factor > 0 -> 重み増加
        # 発火率が高すぎる -> factor < 0 -> 重み減少
        scaling_factor = (self.target_rate - self.avg_firing_rate) * self.alpha
        
        # 重みの更新 (Weight * (1 + scaling))
        # unsqueeze(1) で入力次元に対してブロードキャスト
        with torch.no_grad():
            weight.add_(weight * scaling_factor.unsqueeze(1))


class STDP(SynapseDynamics):
    """Spike-Timing Dependent Plasticity (STDP)"""
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
    """Short-Term Plasticity (STP)"""
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