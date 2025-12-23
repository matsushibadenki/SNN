# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (高精度・恒常性安定版)
# 目的: Acc 50% 超の知能構造を、15% 前後の安定した密度で固定し、明滅（リセット）を防ぐ。

import torch
import torch.nn as nn
from typing import cast

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 初期状態: 疎な状態から開始
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold - 2.0)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.15 # 目標密度 15%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        # 側方抑制の微調整
        if current.max() > 0:
            current = current - (current.mean() * 0.7)
            
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 高精度状態を「低密度」で維持する精密学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正1: 密度依存の成長抑制 (Density-Dependent Growth)
            # 密度が目標(15%)に近づくほど、LTP(強化)の効きを弱くする
            growth_inhibition = max(0.01, 1.0 - (conn_rate / self.target_conn_rate))
            
            if modulation > 0:
                # 成功時: 密度に余裕がある時だけ強く成長、余裕がなければ維持のみ
                self.states.add_(trace * modulation * 10.0 * growth_inhibition)
            else:
                # 失敗時: 密度に関わらず、間違った配線を削る
                self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正2: 指数関数的な剪定圧を「マイルド」に (急激な 0% への転落を防ぐ)
            if conn_rate > self.target_conn_rate:
                # 緩やかな減衰
                self.states.sub_(0.5 * (conn_rate / self.target_conn_rate))
            elif conn_rate < 0.05:
                # 密度が低すぎる時だけ救済
                self.states.add_(0.2)

            self.states.clamp_(1, self.max_states)
