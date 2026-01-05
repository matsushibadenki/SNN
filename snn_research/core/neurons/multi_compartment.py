# ファイルパス: snn_research/core/neurons/multi_compartment.py
# Title: Two-Compartment LIF Neuron with Bursting Dynamics
# Description:
#   細胞体(Soma)と樹状突起(Dendrite)を持つ多区画モデル。
#   修正: Dendritic Plateau Potentialによる「バースト発火(Bursting)」を実装。
#   樹状突起が強く活性化すると、細胞体の閾値が一時的に低下し、高周波発火を誘発する。

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

from spikingjelly.activation_based import base, surrogate # type: ignore

class TwoCompartmentLIF(base.MemoryModule):
    """
    能動的樹状突起とバースト発火機能を持つ2区画LIFニューロン。
    
    Dynamics Enhancement:
    - Bursting: V_d > nmda_threshold の時、somaの閾値を下げる、
      あるいは追加の注入電流 (I_burst) を発生させることでバーストを表現。
    """
    
    v_s: Optional[torch.Tensor]
    v_d: Optional[torch.Tensor]
    burst_state: Optional[torch.Tensor] # バースト状態管理用
    spikes: torch.Tensor

    def __init__(
        self,
        features: int,
        tau_soma: float = 20.0,
        tau_dend: float = 10.0,
        g_coupling: float = 0.3, 
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        nmda_gain: float = 0.5,    
        nmda_threshold: float = 0.8,
        burst_facilitation: float = 0.2, # バースト時の閾値低下量
        surrogate_function: Optional[nn.Module] = None
    ):
        super().__init__()
        self.features = features
        self.tau_soma = tau_soma
        self.tau_dend = tau_dend
        self.g_coupling = g_coupling
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.nmda_gain = nmda_gain
        self.nmda_threshold = nmda_threshold
        self.burst_facilitation = burst_facilitation
        
        self.surrogate_function = surrogate_function if surrogate_function else surrogate.ATan()
        
        self.register_buffer("v_s", None)
        self.register_buffer("v_d", None)
        self.register_buffer("burst_state", None)
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
        self.burst_state = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def _nmda_current(self, v_d: torch.Tensor) -> torch.Tensor:
        """樹状突起におけるNMDA電流（非線形増幅）"""
        activation = torch.sigmoid((v_d - self.nmda_threshold) * 5.0)
        return self.nmda_gain * activation * v_d

    def forward(self, input_soma: torch.Tensor, input_dend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.stateful:
            self.v_s = None
            self.v_d = None
            self.burst_state = None
            
        if self.v_s is None:
            self.v_s = torch.zeros_like(input_soma)
        if self.v_d is None:
            self.v_d = torch.zeros_like(input_dend)
        if self.burst_state is None:
            self.burst_state = torch.zeros_like(input_soma)
            
        decay_s = math.exp(-1.0 / self.tau_soma)
        decay_d = math.exp(-1.0 / self.tau_dend)
        
        # 1. Dendrite Update
        i_nmda = self._nmda_current(self.v_d)
        i_s2d = self.g_coupling * (self.v_s - self.v_d)
        
        # 樹状突起電位の更新
        self.v_d = self.v_d * decay_d + input_dend + i_s2d + i_nmda
        
        # 2. Bursting Logic
        # 樹状突起が強く発火（プラトー）している場合、バースト状態をONにする
        is_plateau = (self.v_d > self.nmda_threshold).float()
        # バースト状態は少し持続する (Time constant decay)
        self.burst_state = self.burst_state * 0.8 + is_plateau * 0.2
        
        # バースト中は実効閾値が下がる
        effective_threshold = self.v_threshold - (self.burst_state * self.burst_facilitation)
        
        # 3. Soma Update
        i_d2s = self.g_coupling * (self.v_d - self.v_s)
        # バースト電流（樹状突起からの強力な流入を模倣）
        i_burst = self.burst_state * 0.1 
        
        self.v_s = self.v_s * decay_s + input_soma + i_d2s + i_burst
        
        # 4. Spike Generation
        spike = self.surrogate_function(self.v_s - effective_threshold)
        
        # 5. Reset
        self.v_s = self.v_s * (1.0 - spike) + self.v_reset * spike
        
        self.spikes = spike
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
            
        return spike, self.v_s