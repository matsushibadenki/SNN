# ファイルパス: snn_research/core/neurons/__init__.py
# (修正: ScaleAndFireNeuronの3次元入力対応)
# Title: SNNニューロンモデル定義
# Description:
# - 4次元入力 (B, C, H, W) などの場合、チャネル次元(dim=1)に合わせてパラメータを
#   (1, C, 1, 1) に変形する _view_params ヘルパーを追加。
# - AdaptiveLIFNeuron, ProbabilisticLIFNeuron, DualThresholdNeuron, GLIFNeuron, TC_LIF に適用。
# - 【修正】ScaleAndFireNeuron が (B, L, C) の3次元入力を正しく処理できるように分岐を追加。

from typing import Optional, Tuple, Any, List, cast
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]

from .bif_neuron import BistableIFNeuron
from .feel_neuron import EvolutionaryLeakLIF # 追加

__all__ = [
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "ProbabilisticLIFNeuron",
    "GLIFNeuron",
    "TC_LIF",
    "DualThresholdNeuron",
    "ScaleAndFireNeuron",
    "BistableIFNeuron",
    "EvolutionaryLeakLIF" # 公開
]

class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    """
    log_tau_mem: nn.Parameter

    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
        noise_intensity: float = 0.0,
        threshold_decay: float = 0.99,
        threshold_step: float = 0.05,
    ):
        super().__init__()
        self.features = features
        
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        self.threshold_decay = threshold_decay
        self.threshold_step = threshold_step
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.adaptive_threshold = None
        self.spikes.zero_()
        self.total_spikes.zero_()
    
    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソル x の形状に合わせてパラメータをリシェイプする。
        特に (B, C, H, W) のような画像入力において、チャネル次元 C にパラメータを合わせる。
        """
        if param.ndim != 1:
            return param
            
        # Conv2d出力 (B, C, H, W) -> param (C) を (1, C, 1, 1) に変形
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
            
        # Conv1d出力 (B, C, L) -> param (C) を (1, C, 1) に変形
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
            
        # RNN/Transformer (B, T, C) の場合は標準ブロードキャスト (..., C) で動作するため何もしない
        
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes one timestep of input current."""
        if not self.stateful:
            self.mem = None
            self.adaptive_threshold = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)

        # --- ▼ 修正: パラメータのブロードキャスト対応 ▼ ---
        log_tau = self._view_params(self.log_tau_mem, x)
        base_thresh = self._view_params(self.base_threshold, x)

        current_tau_mem = torch.exp(log_tau) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        self.mem = self.mem * mem_decay + x
        # --- ▲ 修正 ▲ ---
        
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity
        
        self.adaptive_threshold = self.adaptive_threshold * self.threshold_decay
        current_threshold = base_thresh + self.adaptive_threshold
        spike = self.surrogate_function(self.mem - current_threshold)
        
        # 統計用スパイクカウント（バッチ・空間平均）
        if spike.ndim > 1:
            # (B, C, ...) -> (C,)
            if x.ndim == 4: # (B, C, H, W)
                 self.spikes = spike.mean(dim=(0, 2, 3))
            elif x.ndim == 3 and x.shape[2] == self.features: # (B, T, C)
                 self.spikes = spike.mean(dim=(0, 1))
            elif x.ndim == 3 and x.shape[1] == self.features: # (B, C, L)
                 self.spikes = spike.mean(dim=(0, 2))
            elif x.ndim == 2: # (B, C)
                 self.spikes = spike.mean(dim=0)
            else:
                 self.spikes = spike.mean() # Fallback
        else:
            self.spikes = spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask)
        
        if self.training:
            self.adaptive_threshold = (
                self.adaptive_threshold + self.threshold_step * spike.detach()
            )
        else:
            with torch.no_grad():
                 self.adaptive_threshold = (
                    self.adaptive_threshold + self.threshold_step * spike
                )
        
        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        current_rate = self.spikes.mean()
        target = torch.tensor(self.target_spike_rate, device=current_rate.device)
        return F.mse_loss(current_rate, target)

class IzhikevichNeuron(base.MemoryModule):
    """
    Izhikevich neuron model. (Scalar params, broadcast works automatically)
    """
    def __init__(
        self,
        features: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 0.5,
    ):
        super().__init__()
        self.features = features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.v_peak = 30.0
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.stateful = False

        self.register_buffer("v", None)
        self.register_buffer("u", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.v = None
        self.u = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.stateful:
            self.v = None
            self.u = None
            
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, float(self.c))
        if self.u is None or self.u.shape != x.shape:
            self.u = torch.full_like(x, float(self.b * self.c))

        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt
        
        spike = self.surrogate_function(self.v - self.v_peak)
        
        if spike.ndim > 1:
            # 簡易集計
            self.spikes = spike.mean(dim=0) if spike.shape[0] > 0 else spike
        else:
            self.spikes = spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        
        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, torch.full_like(self.v, float(self.c)), self.v)
        self.u = torch.where(reset_mask, self.u + self.d, self.u)
        
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v
        
class ProbabilisticLIFNeuron(base.MemoryModule):
    """
    Probabilistic LIF Neuron.
    """
    log_tau_mem: nn.Parameter
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        temperature: float = 0.5,
        noise_intensity: float = 0.0,
    ):
        super().__init__()
        self.features = features
        
        initial_log_tau = torch.full((features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        
        self.threshold = threshold
        self.temperature = temperature
        self.noise_intensity = noise_intensity

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()
        
    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1: return param
        if x.ndim == 4 and x.shape[1] == self.features: return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features: return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.stateful:
            self.mem = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)

        # --- ▼ 修正: パラメータのブロードキャスト対応 ▼ ---
        log_tau = self._view_params(self.log_tau_mem, x)
        current_tau_mem = torch.exp(log_tau) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        # --- ▲ 修正 ▲ ---
        
        self.mem = self.mem * mem_decay + x

        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity

        spike_prob = torch.sigmoid((self.mem - self.threshold) / self.temperature)
        spike = (torch.rand_like(self.mem) < spike_prob).float()

        self.spikes = spike.mean() # 簡易

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach()
        self.mem = self.mem * (1.0 - reset_mask)

        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.spikes.device)

class GLIFNeuron(base.MemoryModule):
    """
    Gated Leaky Integrate-and-Fire (GLIF) ニューロン。
    """
    base_threshold: nn.Parameter
    gate_tau_lin: nn.Linear
    v_reset: nn.Parameter

    def __init__(
        self,
        features: int,
        base_threshold: float = 1.0,
        gate_input_features: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        
        if gate_input_features is None:
            gate_input_features = features
        
        self.v_reset = nn.Parameter(torch.full((features,), 0.0))
        
        self.gate_tau_lin = nn.Linear(gate_input_features, features)
        
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()
        
    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1: return param
        if x.ndim == 4 and x.shape[1] == self.features: return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features: return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.stateful:
            self.mem = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        
        # ゲート入力の次元チェックと射影
        if x.shape[1] != self.gate_tau_lin.in_features:
             if self.gate_tau_lin.in_features == self.features:
                 gate_input = x
             else:
                 if not hasattr(self, 'gate_input_proj'):
                     self.gate_input_proj = nn.Linear(x.shape[1], self.gate_tau_lin.in_features).to(x.device)
                 gate_input = self.gate_input_proj(x) # type: ignore[attr-defined]
        else:
             gate_input = x
        
        # ゲート計算 (4D入力の場合は Permute して Linear)
        if gate_input.ndim > 2:
            # (B, C, H, W) -> (B, H, W, C)
            gate_input_perm = gate_input.permute(0, 2, 3, 1)
            mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input_perm)).permute(0, 3, 1, 2)
        else:
            mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input))
        
        # --- ▼ 修正: パラメータのブロードキャスト対応 ▼ ---
        v_reset_gated = self._view_params(self.v_reset, x)
        base_thresh = self._view_params(self.base_threshold, x)
        # --- ▲ 修正 ▲ ---

        self.mem = self.mem * mem_decay_gate + (1.0 - mem_decay_gate) * x
        
        spike = self.surrogate_function(self.mem - base_thresh)
        
        self.spikes = spike.mean()
            
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask) + reset_mask * v_reset_gated
        
        return spike, self.mem

class TC_LIF(base.MemoryModule):
    """
    Two-Compartment LIF (TC-LIF) ニューロン。
    """
    log_tau_s: nn.Parameter
    log_tau_d: nn.Parameter
    w_ds: nn.Parameter
    w_sd: nn.Parameter
    base_threshold: nn.Parameter
    v_reset: nn.Parameter

    def __init__(
        self,
        features: int,
        tau_s_init: float = 5.0,
        tau_d_init: float = 20.0,
        w_ds_init: float = 0.5,
        w_sd_init: float = 0.1,
        base_threshold: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        
        self.log_tau_s = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_s_init - 1.1))))
        self.log_tau_d = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_d_init - 1.1))))
        
        self.w_ds = nn.Parameter(torch.full((features,), w_ds_init))
        self.w_sd = nn.Parameter(torch.full((features,), w_sd_init))

        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("v_s", None)
        self.register_buffer("v_d", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.v_s = None
        self.v_d = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1: return param
        if x.ndim == 4 and x.shape[1] == self.features: return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features: return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.stateful:
            self.v_s = None
            self.v_d = None

        if self.v_s is None or self.v_s.shape != x.shape:
            self.v_s = torch.zeros_like(x)
        if self.v_d is None or self.v_d.shape != x.shape:
            self.v_d = torch.zeros_like(x)

        # --- ▼ 修正: パラメータのブロードキャスト対応 ▼ ---
        log_tau_s = self._view_params(self.log_tau_s, x)
        log_tau_d = self._view_params(self.log_tau_d, x)
        w_ds = self._view_params(self.w_ds, x)
        w_sd = self._view_params(self.w_sd, x)
        base_thresh = self._view_params(self.base_threshold, x)
        v_res = self._view_params(self.v_reset, x)
        # --- ▲ 修正 ▲ ---

        current_tau_s = torch.exp(log_tau_s) + 1.1
        decay_s = torch.exp(-1.0 / current_tau_s)
        
        current_tau_d = torch.exp(log_tau_d) + 1.1
        decay_d = torch.exp(-1.0 / current_tau_d)

        # フィードバック項: 簡易的に w_sd * x と同等の形状に拡張する処理は省略
        dendritic_input = x 
        self.v_d = self.v_d * decay_d + dendritic_input
        
        somatic_input = x + w_ds * self.v_d
        self.v_s = self.v_s * decay_s + somatic_input
        
        spike = self.surrogate_function(self.v_s - base_thresh)
        
        self.spikes = spike.detach()
            
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach()
        self.v_s = self.v_s * (1.0 - reset_mask) + reset_mask * v_res
        
        return spike, self.v_s

class DualThresholdNeuron(base.MemoryModule):
    """
    Dual Threshold Neuron.
    """
    log_tau_mem: nn.Parameter
    threshold_high: nn.Parameter
    threshold_low: nn.Parameter
    v_reset: nn.Parameter
    
    spikes: Tensor
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold_high_init: float = 1.0,
        threshold_low_init: float = 0.5,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.features = features
        self.log_tau_mem = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_mem - 1.1))))
        
        self.threshold_high = nn.Parameter(torch.full((features,), threshold_high_init))
        self.threshold_low = nn.Parameter(torch.full((features,), threshold_low_init))
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1: return param
        if x.ndim == 4 and x.shape[1] == self.features: return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features: return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.stateful:
            self.mem = None

        # --- ▼ 修正: パラメータのブロードキャスト対応 ▼ ---
        t_low = self._view_params(self.threshold_low, x)
        t_high = self._view_params(self.threshold_high, x)
        v_res = self._view_params(self.v_reset, x)
        log_tau = self._view_params(self.log_tau_mem, x)
        # --- ▲ 修正 ▲ ---

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = (t_low.detach() / 2.0).expand_as(x)

        current_tau_mem = torch.exp(log_tau) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        self.mem = self.mem * mem_decay + x
        
        spike_untyped = self.surrogate_function(self.mem - t_high)
        spike: Tensor = cast(Tensor, spike_untyped)
        
        current_spikes_detached: Tensor = spike.detach()
        
        self.spikes = current_spikes_detached

        with torch.no_grad():
            self.total_spikes += current_spikes_detached.sum() # type: ignore[has-type]
        
        reset_mem = self.mem - current_spikes_detached * t_high
        below_low_threshold = reset_mem < t_low
        reset_condition = (current_spikes_detached > 0.5) | below_low_threshold
        
        self.mem = torch.where(
            reset_condition,
            v_res.expand_as(self.mem),
            reset_mem
        )
        
        return spike, self.mem

class ScaleAndFireNeuron(base.MemoryModule):
    """
    Scale-and-Fire (SFN) ニューロン. (Stateful logic unchanged as it is usually for dense layers)
    """
    def __init__(
        self,
        features: int,
        num_levels: int = 8,
        base_threshold: float = 1.0,
    ):
        super().__init__()
        self.features = features
        self.num_levels = num_levels
        
        thresholds = torch.linspace(0.5, num_levels - 0.5, num_levels) / num_levels * base_threshold
        self.thresholds = nn.Parameter(thresholds.unsqueeze(0).repeat(features, 1)) # (Features, K)
        
        self.scales = nn.Parameter(torch.ones(features, num_levels)) # (Features, K)
        
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool):
        pass

    def reset(self):
        super().reset()
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """T=1 processing"""
        
        # Case 1: 4D Input (B, C, H, W)
        if x.ndim == 4:
             B, C, H, W = x.shape
             if C != self.features:
                  raise ValueError(f"Channel mismatch: Input {C} vs Neuron {self.features}")
             
             x_expanded = x.unsqueeze(-1) # (B, C, H, W, 1)
             
             # (C, K) -> (1, C, 1, 1, K)
             thresholds_expanded = self.thresholds.view(1, C, 1, 1, self.num_levels)
             scales_expanded = self.scales.view(1, C, 1, 1, self.num_levels)
             
             spatial_spikes = self.surrogate_function(x_expanded - thresholds_expanded)
             output_analog = (spatial_spikes * scales_expanded).sum(dim=-1) # (B, C, H, W)
             
             self.spikes = spatial_spikes.mean(dim=(0, 2, 3, 4)) 
             with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
                
             return output_analog, output_analog

        # Case 2: 3D Input (B, L, C) - Transformer Sequence
        elif x.ndim == 3:
             B, L, C = x.shape
             if C != self.features:
                  raise ValueError(f"Feature mismatch: Input {C} vs Neuron {self.features}")

             x_expanded = x.unsqueeze(-1) # (B, L, C, 1)
             
             # (C, K) -> (1, 1, C, K)
             thresholds_expanded = self.thresholds.view(1, 1, C, self.num_levels)
             scales_expanded = self.scales.view(1, 1, C, self.num_levels)
             
             spatial_spikes = self.surrogate_function(x_expanded - thresholds_expanded)
             output_analog = (spatial_spikes * scales_expanded).sum(dim=-1) # (B, L, C)
             
             self.spikes = spatial_spikes.mean(dim=(0, 1, 3)) 
             with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
                
             return output_analog, output_analog

        # Case 3: 2D Input (B, C)
        elif x.ndim == 2:
            B, N = x.shape
            if N != self.features:
                raise ValueError(f"Input dimension ({N}) does not match num_neurons ({self.features})")
    
            x_expanded = x.unsqueeze(-1) # (B, N, 1)
            thresholds_expanded = self.thresholds.unsqueeze(0) # (1, N, K)
            
            spatial_spikes = self.surrogate_function(x_expanded - thresholds_expanded)
            
            scales_expanded = self.scales.unsqueeze(0) # (1, N, K)
            
            output_analog = (spatial_spikes * scales_expanded).sum(dim=-1)
            
            self.spikes = spatial_spikes.mean(dim=(0, 2))
            with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
    
            return output_analog, output_analog
        
        else:
             raise ValueError(f"Unsupported input shape: {x.shape}")


__all__ = [
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "ProbabilisticLIFNeuron",
    "GLIFNeuron",
    "TC_LIF",
    "DualThresholdNeuron",
    "ScaleAndFireNeuron",
    "BistableIFNeuron"
]