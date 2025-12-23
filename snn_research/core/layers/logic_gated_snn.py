# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (段階的蒸留・活動保証版)
# 目的: Acc 80%に達するまでは結合を維持し、成功した後にのみ贅肉を削ぎ落とす「知能の彫刻」を完遂する。

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
        
        # 修正1: 最初は全結合からスタートし、徐々に彫り出す
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold + 5)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('success_memory', torch.zeros(1)) # 過去の成功率

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正2: 活動駆動 (ニューロンが黙り込まないよう、低活動時にノイズ注入)
        if current.max() < 0.1:
            current = current + torch.randn_like(current).abs() * 0.5
            
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.85).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率ホメオスタシス (沈黙を破るために閾値を下限0.1まで許容)
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(0.1, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 成功を記憶し、段階的にスパース化する学習則 """
        with torch.no_grad():
            self.success_memory.copy_(self.success_memory * 0.99 + (1.0 if reward > 0 else 0.0) * 0.01)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正3: 成功率に応じたプルーニング圧
            # 成功率(success_memory)が低い間は、結合を維持する。成功し始めたら削る。
            pruning_gate = torch.clamp(self.success_memory * 2.0, 0.1, 1.0).item()
            
            if modulation > 0:
                # 成功時: トレース箇所を強力に固定
                self.states.add_(trace * modulation * 10.0)
            else:
                # 失敗時: 痕跡箇所を削除
                self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正4: 段階的な密度調整
            target_rate = 0.15
            if conn_rate > target_rate:
                # 成功ゲートが開いている時のみ、積極的に削る
                self.states.sub_(0.1 * pruning_gate * (conn_rate / target_rate))
            elif conn_rate < 0.05:
                # 密度が低すぎる場合は緊急浮上
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
