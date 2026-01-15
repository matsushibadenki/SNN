# snn_research/core/neurons/lif_neuron.py
import torch
import torch.nn as nn
from typing import Tuple, Optional

class ATanSurrogate(torch.autograd.Function):
    """
    Spike generation with surrogate gradient for backpropagation.
    """
    @staticmethod
    def forward(ctx, input, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = ctx.alpha / (2 * (1 + (torch.pi / 2 * ctx.alpha * input).pow(2)))
        return grad_input * grad, None

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model (PyTorch Native).
    Implements Current-Injection model: V[t] = V[t-1] * decay + Input[t]
    
    Attributes:
        features (int): Number of neurons.
        tau_mem (float): Membrane time constant.
        v_threshold (float): Voltage threshold for spiking.
        dt (float): Simulation time step.
    """
    def __init__(self, features: int, tau_mem: float = 10.0, v_threshold: float = 1.0, dt: float = 1.0):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.dt = dt
        
        # State
        self.membrane_potential: Optional[torch.Tensor] = None
        self.stateful = False

    def set_stateful(self, value: bool):
        """Set whether the neuron maintains state across forward calls."""
        self.stateful = value
        if not value:
            self.reset()

    def reset(self):
        """Reset membrane potential."""
        self.membrane_potential = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LIF dynamics.
        
        Args:
            x (torch.Tensor): Input current tensor of shape (Batch, Features)
            
        Returns:
            spike (torch.Tensor): Binary spike tensor (1.0 or 0.0)
            mem (torch.Tensor): Membrane potential AFTER update
        """
        # Initialize membrane potential if needed
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)
        elif self.membrane_potential.shape != x.shape:
            # Handle batch size change or reset on shape mismatch
            self.membrane_potential = torch.zeros_like(x)

        # LIF Dynamics: Current Injection Model
        # V[t] = V[t-1] * (1 - dt/tau) + Input * dt
        decay = 1.0 - (self.dt / self.tau_mem)
        decay = max(0.0, min(1.0, decay))
        
        # Update membrane potential
        new_mem = self.membrane_potential * decay + x
        
        # Spike generation (Heaviside step function with Surrogate Gradient)
        spike = ATanSurrogate.apply(new_mem - self.v_threshold)
        
        # Hard Reset: If spiked, reset potential to 0
        new_mem = new_mem * (1.0 - spike)
        
        # Update state if stateful
        if self.stateful:
            self.membrane_potential = new_mem.detach() # Detach
        else:
            self.membrane_potential = None

        return spike, new_mem