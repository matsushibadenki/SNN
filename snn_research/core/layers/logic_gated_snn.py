# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (リソース競合・洗練版)
# 目的: 結合率 100% を物理的に禁止し、Acc 50% 超を 10% 前後のスパース結合で実現する。

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
        
        # 修正1: 目標結合率を 10% に固定
        self.target_conn_rate = 0.10

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 修正2: 側方抑制 (誰かが発火したら他を強く抑制)
        current = torch.matmul(x, w.t()).view(-1)
        if current.max() > 0:
            current = current - (current.mean() * 0.9) # 競争を激化
            
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.6).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正3: トレースを「正解時のみ」有効化するための準備
            self.eligibility_trace.mul_(0.7).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火頻度の恒常性
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(1.0, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 厳格なリソース配分による『情報の彫刻』 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正4: 報酬の非線形性強化 (Acc 48% の成功を逃さない)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            if modulation > 0:
                # 成功時: 結合率が低い時ほど、強く成長させる
                growth_gain = max(0.1, (self.target_conn_rate / (conn_rate + 0.01)))
                self.states.add_(trace * modulation * 10.0 * growth_gain)
            else:
                # 失敗時: 痕跡のある場所を優先的に削る
                self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正5: 物理的な「重みの定員制」
            # 10% を超えた瞬間に、報酬に関わらず全配線を一律に削る
            if conn_rate > self.target_conn_rate:
                # 指数関数的な剪定圧
                decay_power = torch.exp(torch.tensor((conn_rate - self.target_conn_rate) * 10)).item()
                self.states.sub_(0.2 * decay_power)
            
            # 死滅防止
            if conn_rate < 0.02:
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
