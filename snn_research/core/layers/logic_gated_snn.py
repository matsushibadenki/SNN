# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (慣性・固化版)
# 目的: 結合率の激しい振動(2-100%)を抑え、Acc 20%超の知能を恒久的なスパース構造として定着させる。

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
        
        # 初期状態
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 10, self.threshold + 2, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正1: 結合の「固さ」を管理するバッファ (Consolidation)
        self.register_buffer('synaptic_firmness', torch.zeros(out_features, in_features))
        self.target_conn_rate = 0.12 # 目標 12%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 修正2: 入力依存の抑制を加え、過剰発火を物理的に防ぐ
        v_mem.mul_(0.7).add_(current / (current.mean() + 1.0))
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 4.0)
            
            # 恒常性
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(1.0, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 慣性と固化を伴う自己組織化ルール """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正3: 摩擦係数の導入 (結合率が高いほど、増える方向の動きにブレーキ)
            friction = torch.exp(torch.tensor(conn_rate / self.target_conn_rate)).item()
            
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            if modulation > 0:
                # 成功時: トレース箇所を強化し、同時に「固さ」を増す
                update = (trace * modulation * 10.0) / friction
                self.states.add_(update)
                self.synaptic_firmness.add_(trace * 0.1)
            else:
                # 失敗時: 「固まっていない」配線を優先的に削る
                decay = (trace * abs(modulation) * 5.0) * (1.0 - torch.tanh(self.synaptic_firmness))
                self.states.sub_(decay)
            
            # 修正4: 基礎代謝の適応化 (固まった配線は削れにくい)
            metabolic_decay = 0.1 * (1.0 - torch.tanh(self.synaptic_firmness))
            self.states.sub_(metabolic_decay)
            
            # 過密時の緊急剪定
            if conn_rate > 0.30:
                self.states.sub_(2.0)
            
            # 死滅防止
            if conn_rate < 0.03:
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
            self.synaptic_firmness.mul_(0.99) # 時間とともに少しずつ柔らかくなる
