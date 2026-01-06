# ファイルパス: snn_research/core/neurons/lif_neuron.py
# Title: Standard LIF Neuron (Autograd Compatible)
# Description:
#   代理勾配(Surrogate Gradient)に対応した標準的なLIFニューロンモデル。
#   core/neurons/bif_neuron.py とインターフェースを統一。

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

# 代理勾配デフォルト


class ATanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (1 + (ctx.alpha * input).pow(2)) * ctx.alpha, None


def atan(alpha=2.0):
    def func(x):
        return ATanSurrogate.apply(x, alpha)
    return func


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron
    dU/dt = -(U - U_rest)/tau + I
    """

    def __init__(self,
                 features: int,
                 tau_mem: float = 20.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 surrogate_function: Callable = atan()):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        self.surrogate_function = surrogate_function

        self.membrane_potential: Optional[torch.Tensor] = None
        self.spikes: Optional[torch.Tensor] = None

        # Stateful compatible interface
        self.stateful = False

    def reset(self):
        self.membrane_potential = None
        self.spikes = None

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.membrane_potential is None or self.membrane_potential.shape != x.shape:
            self.membrane_potential = torch.full_like(x, self.v_rest)

        # Dynamics
        # V[t] = V[t-1] + dt/tau * (-(V[t-1] - V_rest) + x[t])

        mem = self.membrane_potential
        leak = (mem - self.v_rest) / self.tau_mem
        mem_next = mem + self.dt * (-leak + x)

        # Spike
        spike = self.surrogate_function(mem_next - self.v_threshold)

        # Reset (Hard Reset)
        mem_next = torch.where(spike > 0.5, torch.full_like(
            mem_next, self.v_reset), mem_next)

        self.membrane_potential = mem_next
        self.spikes = spike

        return spike, self.membrane_potential
