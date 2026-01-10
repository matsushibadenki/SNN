# snn_research/core/neurons/lif_neuron.py
# Title: Advanced LIF Neuron (Spatiotemporal Ready)
# Description:
#   Stepモード(RNN的利用)とMulti-stepモード(Transformer的利用)の両方をサポートするLIF。
#   サロゲート勾配の選択が可能になり、入力形状 (T, B, C...) に対応。
#   修正: 戻り値を(spike, mem)のタプルに変更し、set_statefulメソッドを追加。

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ..surrogates import surrogate_factory


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with Spatiotemporal Support

    Args:
        tau_mem (float): 膜電位の時定数
        detatch_reset (bool): リセット時の勾配を切断するかどうか
        step_mode (str): 's' (single-step) or 'm' (multi-step)
        surrogate_name (str): サロゲート関数の種類 ('atan', 'sigmoid', 'piecewise')
        surrogate_alpha (float): サロゲート関数の鋭さパラメータ
    """

    def __init__(self,
                 features: int = 0,  # 互換性のため残すが、基本は動的に形状決定
                 tau_mem: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 detach_reset: bool = True,
                 step_mode: str = 's',
                 surrogate_name: str = 'atan',
                 surrogate_alpha: float = 2.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.is_stateful = True

        # サロゲート関数の生成
        self.surrogate_function = surrogate_factory(
            surrogate_name, surrogate_alpha)

        # 内部状態
        self.membrane_potential: Optional[torch.Tensor] = None
        self.spikes: Optional[torch.Tensor] = None

    def set_stateful(self, stateful: bool = True):
        """
        状態保持モードの設定 (SpikingJelly等とのAPI互換性のため)
        """
        self.is_stateful = stateful

    def reset(self):
        """内部状態のリセット"""
        self.membrane_potential = None
        self.spikes = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass

        Input:
            step_mode='s': (Batch, Features...)
            step_mode='m': (Time, Batch, Features...)
        Output:
            (spikes, membrane_potential)
        """
        if self.step_mode == 's':
            return self._forward_step(x)
        elif self.step_mode == 'm':
            return self._forward_multistep(x)
        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

    def _forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """単一タイムステップの処理"""
        if self.membrane_potential is None or self.membrane_potential.shape != x.shape:
            self.membrane_potential = torch.full_like(x, self.v_rest)

        # 膜電位の更新 (Euler法)
        # V[t] = V[t-1] + (1/tau) * (-(V[t-1] - V_rest) + X[t])

        # Decay factor calculation: beta = exp(-dt/tau) or linear approx 1 - dt/tau
        decay = 1.0 / self.tau_mem

        mem_prev = self.membrane_potential

        # 積分 (Integrate)
        mem_next = mem_prev * (1.0 - decay) + x * self.dt

        # 発火 (Fire)
        spike = self.surrogate_function(mem_next - self.v_threshold)

        # リセット (Reset)
        if self.detach_reset:
            spike_for_reset = spike.detach()
        else:
            spike_for_reset = spike

        # Hard Reset: 発火したら v_reset に戻す
        mem_next = mem_next * (1.0 - spike_for_reset) + \
            self.v_reset * spike_for_reset

        self.membrane_potential = mem_next
        self.spikes = spike

        return spike, self.membrane_potential

    def _forward_multistep(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        複数タイムステップの一括処理 (Time, Batch, ...)
        """
        T = x_seq.shape[0]
        spike_seq = []
        mem_seq = []

        # 初期状態
        if self.membrane_potential is None:
            mem = torch.full_like(x_seq[0], self.v_rest)
        else:
            mem = self.membrane_potential

        decay = 1.0 / self.tau_mem

        # 時間軸ループ
        for t in range(T):
            x = x_seq[t]

            # Integrate
            mem = mem * (1.0 - decay) + x * self.dt

            # Fire
            spike = self.surrogate_function(mem - self.v_threshold)

            # Reset
            if self.detach_reset:
                spike_reset = spike.detach()
            else:
                spike_reset = spike

            mem = mem * (1.0 - spike_reset) + self.v_reset * spike_reset

            spike_seq.append(spike)
            mem_seq.append(mem)

        # 最終状態を保存
        self.membrane_potential = mem

        # Stackして返す (Time, Batch, ...)
        # 戻り値は (Spikes, Membrane_Potentials)
        return torch.stack(spike_seq, dim=0), torch.stack(mem_seq, dim=0)
