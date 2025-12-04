# ファイルパス: snn_research/core/neurons/bif_neuron.py
# Title: Bistable Integrate-and-Fire (BIF) ニューロンモデル (互換性修正版)
# Description: 
#   双安定性を持ち、確率的な挙動を示す積分発火ニューロン。
#   修正点:
#   - forwardメソッドの戻り値を (spike, mem) のタプルに変更し、
#     AdaptiveLIFNeuron など他のニューロンクラスとの互換性を確保。
#     (AdaptiveNeuronSelectorでの動的切り替え時にクラッシュするのを防ぐため)

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple, Dict, Any, Union
import logging

from spikingjelly.activation_based import surrogate, base  # type: ignore

try:
    import numpy as np
    import matplotlib.pyplot as plt
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    np = None # type: ignore
    plt = None # type: ignore
    _VISUALIZATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BistableIFNeuron(base.MemoryModule):
    """
    双安定積分発火ニューロン (Bistable Integrate-and-Fire Neuron)。
    """

    membrane_potential: Optional[torch.Tensor]
    spikes: Optional[torch.Tensor]

    def __init__(self,
                 features: int,
                 v_threshold_high: float = 1.0,
                 v_reset: float = 0.6,
                 tau_mem: float = 10.0,
                 bistable_strength: float = 0.25,
                 v_rest: float = 0.0,
                 unstable_equilibrium_offset: float = 0.5,
                 surrogate_function: Callable = surrogate.ATan(alpha=2.0)
                ):
        super().__init__()
        self.features = features
        self.v_th_high = v_threshold_high
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.dt = 1.0 

        self.bistable_strength = bistable_strength
        self.v_rest = v_rest
        self.unstable_equilibrium = self.v_rest + unstable_equilibrium_offset
        self.surrogate_function = surrogate_function

        self.stateful = False
        self.membrane_potential = None
        self.spikes = None

        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self) -> None:
        super().reset()
        self.membrane_potential = None
        self.spikes = None
        if hasattr(self, 'total_spikes'):
            self.total_spikes.zero_()

    def _bif_dynamics(self, v: torch.Tensor, input_current: torch.Tensor) -> torch.Tensor:
        leak = (v - self.v_rest) / self.tau_mem
        non_linear = self.bistable_strength * (v - self.v_rest) * (v - self.unstable_equilibrium) * (self.v_th_high - v)
        dv = (-leak + non_linear + input_current) * self.dt
        return v + dv

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1タイムステップ分の処理を実行します。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (スパイク, 膜電位)
        """
        if not self.stateful or self.membrane_potential is None or self.membrane_potential.shape != input_current.shape:
            noise = torch.randn_like(input_current) * 0.05
            self.membrane_potential = (noise + self.unstable_equilibrium).to(input_current.device)

        current_mem = self.membrane_potential if self.membrane_potential is not None else torch.zeros_like(input_current)
        next_potential = self._bif_dynamics(current_mem, input_current)

        spike = self.surrogate_function(next_potential - self.v_th_high)

        reset_potential = torch.where(spike > 0.5, torch.full_like(next_potential, self.v_reset), next_potential)

        self.membrane_potential = reset_potential
        self.spikes = spike

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        # --- 修正: (spike, mem) のタプルを返す ---
        return spike, self.membrane_potential

    # simulate_fluctuations メソッドは変更なし (省略)
    def simulate_fluctuations(self,
                              input_sequence: Optional[torch.Tensor] = None,
                              n_steps: int = 1000,
                              batch_shape: Tuple[int, ...] = (1, 1),
                              device: Optional[torch.device] = None,
                              noise_scale: float = 0.05,
                              initial_perturb: float = 0.05,
                              commit: bool = False,
                              return_dict: bool = True
                              ) -> Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        # (元の実装のまま)
        if not _VISUALIZATION_AVAILABLE:
            logging.warning("simulate_fluctuations: numpy or matplotlib not found. Skipping simulation.")
            return {} if return_dict else (torch.tensor([]), torch.tensor([]))

        if device is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        if input_sequence is not None:
            inp = input_sequence.to(device)
            T = inp.shape[0]
        else:
            T = n_steps
            inp_shape = (T, ) + batch_shape
            inp = torch.zeros(*inp_shape, device=device)

        state_shape = inp.shape[1:]
        v0 = (torch.ones(*state_shape, device=device) * self.unstable_equilibrium) + torch.randn(*state_shape, device=device) * initial_perturb

        v = v0.clone()
        spikes_hist = []
        potentials = []

        with torch.no_grad():
            for t in range(T):
                current = inp[t]
                next_v = self._bif_dynamics(v, current)
                next_v = next_v + (noise_scale * torch.randn_like(next_v))
                spike = (next_v >= self.v_th_high).float()
                v = torch.where(spike > 0.5, torch.full_like(next_v, self.v_reset), next_v)
                
                potentials.append(v.cpu().clone())
                spikes_hist.append(spike.cpu().clone())

        potentials_t = torch.stack(potentials, dim=0)
        spikes_t = torch.stack(spikes_hist, dim=0)

        pot_flat = potentials_t[int(T*0.5):].flatten().double()
        
        if pot_flat.numel() > 0:
            rng = float(pot_flat.max().item() - pot_flat.min().item())
            std = float(torch.std(pot_flat).item())
            estimated_levels = max(1.0, (rng / (2.0 * std))) if std > 0 else 1.0
            estimated_bits = float(np.log2(estimated_levels)) if estimated_levels > 0 else 0.0
        else:
            rng, std, estimated_levels, estimated_bits = 0.0, 0.0, 1.0, 0.0

        result = {
            "membrane_potentials": potentials_t,
            "spikes": spikes_t,
            "params": {
                "noise_scale": noise_scale,
                "initial_perturb": initial_perturb,
                "v_th_high": self.v_th_high,
                "v_reset": self.v_reset,
                "bistable_strength": self.bistable_strength
            },
            "stats": {
                "range": rng,
                "std": std,
                "estimated_levels": estimated_levels,
                "estimated_bits": estimated_bits,
                "total_spikes": float(spikes_t.sum().item())
            }
        }

        if commit:
            self.membrane_potential = potentials_t[-1].to(device)
            self.spikes = spikes_t[-1].to(device)
            with torch.no_grad():
                self.total_spikes += torch.tensor(float(spikes_t.sum().item()), device=self.total_spikes.device)

        if return_dict:
            return result
        else:
            return potentials_t, spikes_t

if __name__ == "__main__":
    pass
