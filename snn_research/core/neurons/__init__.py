# ファイルパス: snn_research/core/neurons/__init__.py
# 日本語タイトル: SNNニューロンモデル定義 (High-Fidelity & Adaptive)
# 修正 (v3.2): MPSエラー(Placeholder storage)対策として、ScaleAndFireNeuron等の入力に
#             .contiguous() を強制適用し、メモリ配置を整列させる。

from typing import Optional, Tuple, Any, cast
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
# type: ignore[import-untyped]
from spikingjelly.activation_based import surrogate, base

# 依存モジュールのインポート
from .bif_neuron import BistableIFNeuron
from .feel_neuron import EvolutionaryLeakLIF

__all__ = [
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "ProbabilisticLIFNeuron",
    "GLIFNeuron",
    "TC_LIF",
    "DualThresholdNeuron",
    "ScaleAndFireNeuron",
    "BistableIFNeuron",
    "EvolutionaryLeakLIF"
]


class LearnableATan(torch.autograd.Function):
    """
    学習可能な傾きを持つ ArcTan サロゲート勾配関数。
    """
    @staticmethod
    def forward(ctx: Any, input: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(input, alpha)
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        input, alpha = ctx.saved_tensors
        grad_input = grad_output * \
            ((alpha / 2) / (1 + (torch.pi / 2 * alpha * input).pow(2)))
        return grad_input, None


class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron.
    """
    log_tau_mem: nn.Parameter
    base_threshold: nn.Parameter
    surrogate_alpha: nn.Parameter

    # 状態変数
    avg_firing_rate: torch.Tensor
    mem: Optional[torch.Tensor]
    adaptive_threshold: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor
    refractory_count: Optional[torch.Tensor]

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
        v_reset: float = 0.0,
        homeostasis_rate: float = 0.001,
        refractory_period: int = 2
    ):
        super().__init__()
        self.features = features
        initial_log_tau = torch.full(
            (features,), math.log(max(1.1, tau_mem - 1.1)))
        self.log_tau_mem = nn.Parameter(initial_log_tau)
        self.base_threshold = nn.Parameter(
            torch.full((features,), base_threshold))
        self.surrogate_alpha = nn.Parameter(torch.tensor(2.0))

        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        self.threshold_decay = threshold_decay
        self.threshold_step = threshold_step
        self.v_reset = v_reset
        self.homeostasis_rate = homeostasis_rate
        self.refractory_period = refractory_period

        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        self.register_buffer("refractory_count", None)
        self.register_buffer("avg_firing_rate", torch.zeros(features))
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
        self.refractory_count = None
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        if x.ndim == 3 and x.shape[2] == self.features:
            return param
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # [Fix] MPS Memory Contiguity check
        if not x.is_contiguous():
            x = x.contiguous()

        if not self.stateful:
            self.mem = None
            self.adaptive_threshold = None
            self.refractory_count = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)
        if self.refractory_count is None or self.refractory_count.shape != x.shape:
            self.refractory_count = torch.zeros_like(x)

        log_tau = self._view_params(self.log_tau_mem, x)
        base_thresh = self._view_params(self.base_threshold, x)

        current_tau_mem = torch.exp(log_tau) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)

        is_refractory = (self.refractory_count > 0).float()
        integrated_mem = self.mem * mem_decay + x

        if self.training and self.noise_intensity > 0:
            integrated_mem += torch.randn_like(integrated_mem) * \
                self.noise_intensity

        self.mem = (1.0 - is_refractory) * integrated_mem + \
            is_refractory * self.v_reset

        self.adaptive_threshold = self.adaptive_threshold * self.threshold_decay
        current_threshold = base_thresh + self.adaptive_threshold

        spike_potential = self.mem - current_threshold
        spike = LearnableATan.apply(
            spike_potential, F.softplus(self.surrogate_alpha))
        spike = spike * (1.0 - is_refractory)

        if spike.ndim > 1:
            if x.ndim == 4:
                spike_mean_spatial = spike.mean(dim=(0, 2, 3))
            elif x.ndim == 3 and x.shape[2] == self.features:
                spike_mean_spatial = spike.mean(dim=(0, 1))
            elif x.ndim == 3 and x.shape[1] == self.features:
                spike_mean_spatial = spike.mean(dim=(0, 2))
            elif x.ndim == 2:
                spike_mean_spatial = spike.mean(dim=0)
            else:
                spike_mean_spatial = spike.mean()
        else:
            spike_mean_spatial = spike

        self.spikes = spike_mean_spatial

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
            alpha = 0.01
            self.avg_firing_rate = (
                1 - alpha) * self.avg_firing_rate + alpha * spike_mean_spatial.detach()

            if self.training:
                rate_error = self.avg_firing_rate - self.target_spike_rate
                tolerance = self.target_spike_rate * 0.1
                update_mask = (rate_error.abs() > tolerance).float()
                delta_th = rate_error * self.homeostasis_rate * update_mask
                self.base_threshold.data += delta_th
                self.base_threshold.data.clamp_(min=0.1)

        spike_detached = spike.detach()
        new_refractory = spike_detached * self.refractory_period
        self.refractory_count = torch.clamp(self.refractory_count - 1, min=0.0)
        self.refractory_count = torch.max(
            self.refractory_count, new_refractory)

        self.mem = self.mem * (1.0 - spike_detached) + \
            self.v_reset * spike_detached

        if self.training:
            self.adaptive_threshold = self.adaptive_threshold + \
                self.threshold_step * spike_detached
        else:
            with torch.no_grad():
                self.adaptive_threshold = self.adaptive_threshold + self.threshold_step * spike

        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        current_rate = self.spikes.mean()
        target = torch.tensor(self.target_spike_rate,
                              device=current_rate.device)
        return F.mse_loss(current_rate, target)


class IzhikevichNeuron(base.MemoryModule):
    # (省略なしの実装を維持)
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    v: Optional[torch.Tensor]
    u: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor

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
        self.register_buffer('a', torch.full((features,), a))
        self.register_buffer('b', torch.full((features,), b))
        self.register_buffer('c', torch.full((features,), c))
        self.register_buffer('d', torch.full((features,), d))
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
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not x.is_contiguous(): x = x.contiguous()
        if not self.stateful:
            self.v = None
            self.u = None

        a = self._view_params(self.a, x)
        b = self._view_params(self.b, x)
        c = self._view_params(self.c, x)
        d = self._view_params(self.d, x)

        if self.v is None or self.v.shape != x.shape:
            self.v = c.expand_as(x).clone()
        if self.u is None or self.u.shape != x.shape:
            self.u = (b * self.v).clone()

        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = a * (b * self.v - self.u)

        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt

        spike = self.surrogate_function(self.v - self.v_peak)

        if spike.ndim > 1:
            if x.ndim == 4:
                self.spikes = spike.mean(dim=(0, 2, 3))
            elif x.ndim == 3:
                self.spikes = spike.mean(
                    dim=(0, 1)) if x.shape[2] == self.features else spike.mean(dim=(0, 2))
            else:
                self.spikes = spike.mean(dim=0)
        else:
            self.spikes = spike

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, c.expand_as(self.v), self.v)
        self.u = torch.where(reset_mask, self.u + d, self.u)
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v


class ProbabilisticLIFNeuron(base.MemoryModule):
    log_tau_mem: nn.Parameter
    mem: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor

    def __init__(self, features: int, tau_mem: float = 20.0, threshold: float = 1.0, temperature: float = 0.5, noise_intensity: float = 0.0, v_reset: float = 0.0):
        super().__init__()
        self.features = features
        initial_log_tau = torch.full(
            (features,), math.log(max(1.1, tau_mem - 1.1)))
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
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not x.is_contiguous(): x = x.contiguous()
        if not self.stateful:
            self.mem = None
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        log_tau = self._view_params(self.log_tau_mem, x)
        current_tau_mem = torch.exp(log_tau) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        self.mem = self.mem * mem_decay + x
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity
        spike_prob = torch.sigmoid(
            (self.mem - self.threshold) / self.temperature)
        spike = (torch.rand_like(self.mem) < spike_prob).float()
        self.spikes = spike.mean()
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        reset_mask = spike.detach()
        self.mem = self.mem * (1.0 - reset_mask)
        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor: return torch.tensor(0.0)


class GLIFNeuron(base.MemoryModule):
    base_threshold: nn.Parameter
    gate_tau_lin: nn.Linear
    v_reset: nn.Parameter
    mem: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor

    def __init__(self, features: int, base_threshold: float = 1.0, gate_input_features: Optional[int] = None, **kwargs: Any):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(
            torch.full((features,), base_threshold))
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
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not x.is_contiguous(): x = x.contiguous()
        if not self.stateful:
            self.mem = None
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if x.shape[1] != self.gate_tau_lin.in_features:
            if self.gate_tau_lin.in_features == self.features:
                gate_input = x
            else:
                if not hasattr(self, 'gate_input_proj'):
                    self.gate_input_proj = nn.Linear(
                        x.shape[1], self.gate_tau_lin.in_features).to(x.device)
                gate_input = self.gate_input_proj(x)  # type: ignore
        else:
            gate_input = x
        if gate_input.ndim > 2:
            gate_input_perm = gate_input.permute(0, 2, 3, 1)
            mem_decay_gate = torch.sigmoid(
                self.gate_tau_lin(gate_input_perm)).permute(0, 3, 1, 2)
        else:
            mem_decay_gate = torch.sigmoid(self.gate_tau_lin(gate_input))
        v_reset_gated = self._view_params(self.v_reset, x)
        base_thresh = self._view_params(self.base_threshold, x)
        self.mem = self.mem * mem_decay_gate + (1.0 - mem_decay_gate) * x
        spike = self.surrogate_function(self.mem - base_thresh)
        self.spikes = spike.mean()
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        reset_mask = spike.detach()
        self.mem = self.mem * (1.0 - reset_mask) + reset_mask * v_reset_gated
        return spike, self.mem


class TC_LIF(base.MemoryModule):
    log_tau_s: nn.Parameter
    log_tau_d: nn.Parameter
    w_ds: nn.Parameter
    w_sd: nn.Parameter
    base_threshold: nn.Parameter
    v_reset: nn.Parameter
    v_s: Optional[torch.Tensor]
    v_d: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor

    def __init__(self, features: int, tau_s_init: float = 5.0, tau_d_init: float = 20.0, w_ds_init: float = 0.5, w_sd_init: float = 0.1, base_threshold: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.features = features
        self.base_threshold = nn.Parameter(
            torch.full((features,), base_threshold))
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        self.log_tau_s = nn.Parameter(torch.full(
            (features,), math.log(max(1.1, tau_s_init - 1.1))))
        self.log_tau_d = nn.Parameter(torch.full(
            (features,), math.log(max(1.1, tau_d_init - 1.1))))
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
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not x.is_contiguous(): x = x.contiguous()
        if not self.stateful:
            self.v_s = None
            self.v_d = None
        if self.v_s is None or self.v_s.shape != x.shape:
            self.v_s = torch.zeros_like(x)
        if self.v_d is None or self.v_d.shape != x.shape:
            self.v_d = torch.zeros_like(x)
        log_tau_s = self._view_params(self.log_tau_s, x)
        log_tau_d = self._view_params(self.log_tau_d, x)
        w_ds = self._view_params(self.w_ds, x)

        base_thresh = self._view_params(self.base_threshold, x)
        v_res = self._view_params(self.v_reset, x)
        current_tau_s = torch.exp(log_tau_s) + 1.1
        decay_s = torch.exp(-1.0 / current_tau_s)
        current_tau_d = torch.exp(log_tau_d) + 1.1
        decay_d = torch.exp(-1.0 / current_tau_d)
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
    log_tau_mem: nn.Parameter
    threshold_high: nn.Parameter
    threshold_low: nn.Parameter
    v_reset: nn.Parameter
    mem: Optional[torch.Tensor]
    spikes: torch.Tensor
    total_spikes: torch.Tensor

    def __init__(self, features: int, tau_mem: float = 20.0, threshold_high_init: float = 1.0, threshold_low_init: float = 0.5, v_reset: float = 0.0):
        super().__init__()
        self.features = features
        self.log_tau_mem = nn.Parameter(torch.full(
            (features,), math.log(max(1.1, tau_mem - 1.1))))
        self.threshold_high = nn.Parameter(
            torch.full((features,), threshold_high_init))
        self.threshold_low = nn.Parameter(
            torch.full((features,), threshold_low_init))
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
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if param.ndim != 1:
            return param
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        if x.ndim == 3 and x.shape[1] == self.features:
            return param.view(1, -1, 1)
        return param

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not x.is_contiguous(): x = x.contiguous()
        if not self.stateful:
            self.mem = None
        t_low = self._view_params(self.threshold_low, x)
        t_high = self._view_params(self.threshold_high, x)
        v_res = self._view_params(self.v_reset, x)
        log_tau = self._view_params(self.log_tau_mem, x)
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
            self.total_spikes += current_spikes_detached.sum()
        reset_mem = self.mem - current_spikes_detached * t_high
        below_low_threshold = reset_mem < t_low
        reset_condition = (current_spikes_detached > 0.5) | below_low_threshold
        self.mem = torch.where(
            reset_condition, v_res.expand_as(self.mem), reset_mem)
        return spike, self.mem


class ScaleAndFireNeuron(base.MemoryModule):
    spikes: torch.Tensor
    total_spikes: torch.Tensor

    def __init__(self, features: int, num_levels: int = 8, base_threshold: float = 1.0):
        super().__init__()
        self.features = features
        self.num_levels = num_levels
        thresholds = torch.linspace(
            0.5, num_levels - 0.5, num_levels) / num_levels * base_threshold
        self.thresholds = nn.Parameter(thresholds.unsqueeze(
            0).repeat(features, 1))  # (Features, K)
        self.scales = nn.Parameter(torch.ones(
            features, num_levels))  # (Features, K)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool): pass

    def reset(self):
        super().reset()
        self.spikes = torch.zeros_like(self.spikes)
        self.total_spikes = torch.zeros_like(self.total_spikes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # [Critical Fix for MPS] Ensure memory is contiguous before broadcasting
        if not x.is_contiguous():
            x = x.contiguous()

        if x.ndim == 4:
            B, C, H, W = x.shape
            if C != self.features:
                raise ValueError(
                    f"Channel mismatch: Input {C} vs Neuron {self.features}")
            x_expanded = x.unsqueeze(-1)
            thresholds_expanded = self.thresholds.view(
                1, C, 1, 1, self.num_levels)
            scales_expanded = self.scales.view(1, C, 1, 1, self.num_levels)
            spatial_spikes = self.surrogate_function(
                x_expanded - thresholds_expanded)
            output_analog = (spatial_spikes * scales_expanded).sum(dim=-1)
            self.spikes = spatial_spikes.mean(dim=(0, 2, 3, 4))
            with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
            return output_analog, output_analog
        elif x.ndim == 3:
            B, L, C = x.shape
            if C != self.features:
                raise ValueError(
                    f"Feature mismatch: Input {C} vs Neuron {self.features}")
            x_expanded = x.unsqueeze(-1)
            thresholds_expanded = self.thresholds.view(
                1, 1, C, self.num_levels)
            scales_expanded = self.scales.view(1, 1, C, self.num_levels)
            spatial_spikes = self.surrogate_function(
                x_expanded - thresholds_expanded)
            output_analog = (spatial_spikes * scales_expanded).sum(dim=-1)
            self.spikes = spatial_spikes.mean(dim=(0, 1, 3))
            with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
            return output_analog, output_analog
        elif x.ndim == 2:
            B, N = x.shape
            if N != self.features:
                raise ValueError(
                    f"Input dimension ({N}) does not match num_neurons ({self.features})")
            x_expanded = x.unsqueeze(-1)
            thresholds_expanded = self.thresholds.unsqueeze(0)
            spatial_spikes = self.surrogate_function(
                x_expanded - thresholds_expanded)
            scales_expanded = self.scales.unsqueeze(0)
            output_analog = (spatial_spikes * scales_expanded).sum(dim=-1)
            self.spikes = spatial_spikes.mean(dim=(0, 2))
            with torch.no_grad():
                self.total_spikes += spatial_spikes.detach().sum()
            return output_analog, output_analog
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")