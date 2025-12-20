# ファイルパス: snn_research/core/neurons/multi_compartment.py
# Title: Two-Compartment LIF Neuron with Active Dendrites
# Description:
#   doc/assignment.md 第2章に基づき、細胞体(Soma)と樹状突起(Dendrite)を
#   分離した多区画ニューロンモデルを実装。
#   樹状突起におけるNMDAスパイク（非線形増幅）をシミュレートし、
#   単なる線形加算を超えた計算能力を提供する。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import math

from spikingjelly.activation_based import base, surrogate # type: ignore

class TwoCompartmentLIF(base.MemoryModule):
    """
    能動的樹状突起を持つ2区画LIFニューロンモデル。
    
    Structure:
        - Soma (細胞体): スパイク生成を担当。樹状突起からの入力と直接入力を積分。
        - Dendrite (樹状突起): 入力を積分し、NMDA型非線形応答（プラトー電位）を生成して細胞体へ送る。
    
    Dynamics:
        V_s[t] = V_s[t-1] * decay_s + I_soma + g_c * (V_d - V_s)
        V_d[t] = V_d[t-1] * decay_d + I_dend + g_c * (V_s - V_d) + I_NMDA(V_d)
    """
    
    v_s: Optional[torch.Tensor] # 細胞体膜電位
    v_d: Optional[torch.Tensor] # 樹状突起膜電位
    spikes: torch.Tensor

    def __init__(
        self,
        features: int,
        tau_soma: float = 20.0,
        tau_dend: float = 10.0,
        g_coupling: float = 0.3, # 区画間の結合コンダクタンス
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        nmda_gain: float = 0.5,    # NMDA電流の強さ
        nmda_threshold: float = 0.8, # NMDAスパイク発生の閾値
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
        
        self.surrogate_function = surrogate_function if surrogate_function else surrogate.ATan()
        
        # 状態変数
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

    def _nmda_current(self, v_d: torch.Tensor) -> torch.Tensor:
        """
        樹状突起におけるNMDA電流（非線形増幅）を計算。
        膜電位が閾値を超えると、Mgブロックが外れて大きな電流が流れる現象を模倣。
        """
        # シグモイド関数で閾値以上の電位を滑らかに増幅
        activation = torch.sigmoid((v_d - self.nmda_threshold) * 5.0)
        return self.nmda_gain * activation * v_d

    def forward(self, input_soma: torch.Tensor, input_dend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_soma: 細胞体への直接入力 (Batch, Features) - 例: Feedforward
            input_dend: 樹状突起への入力 (Batch, Features) - 例: Feedback / Context
            
        Returns:
            spike: 出力スパイク
            v_s: 細胞体膜電位
        """
        if not self.stateful:
            self.v_s = None
            self.v_d = None
            
        if self.v_s is None:
            self.v_s = torch.zeros_like(input_soma)
        if self.v_d is None:
            self.v_d = torch.zeros_like(input_dend)
            
        # 減衰係数
        decay_s = math.exp(-1.0 / self.tau_soma)
        decay_d = math.exp(-1.0 / self.tau_dend)
        
        # 1. 樹状突起の更新
        # NMDA非線形項の計算
        i_nmda = self._nmda_current(self.v_d)
        
        # 区画間電流 (Soma -> Dendrite)
        i_s2d = self.g_coupling * (self.v_s - self.v_d)
        
        # V_d 更新: Leak + Input + Coupling + NMDA
        self.v_d = self.v_d * decay_d + input_dend + i_s2d + i_nmda
        
        # 2. 細胞体の更新
        # 区画間電流 (Dendrite -> Soma)
        i_d2s = self.g_coupling * (self.v_d - self.v_s)
        
        # V_s 更新: Leak + Input + Coupling
        self.v_s = self.v_s * decay_s + input_soma + i_d2s
        
        # 3. スパイク生成
        spike = self.surrogate_function(self.v_s - self.v_threshold)
        
        # 4. リセット (Hard Reset)
        self.v_s = self.v_s * (1.0 - spike) + self.v_reset * spike
        # 樹状突起の電位はスパイク後も（バックプロパゲーションでリセットされることもあるが）
        # ここでは文脈保持のためにリセットしない設定とする
        
        self.spikes = spike
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
            
        return spike, self.v_s