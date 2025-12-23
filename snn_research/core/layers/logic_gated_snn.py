# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (生命維持・活動駆動版)
# 目的: Conn: 0.0% による知能の死を物理的に不可能にし、常に情報の探索を継続させる。

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
        
        # 初期状態: 閾値周辺に分布 (一部は最初から結合)
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 3.0 + self.threshold)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 低活動状態からの復帰用カウンタ
        self.register_buffer('silent_steps', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)

        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        # 1. 膜電位更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        # 2. 活動駆動 (Activity Drive): 長く発火していないニューロンにノイズ注入
        drive = (self.silent_steps > 50).float() * torch.randn_like(v_mem).abs() * 2.0
        v_mem.add_(drive)
        
        # 3. 発火判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 4. 適格性トレースの更新
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 沈黙カウンタの更新
            self.silent_steps.add_(1.0).sub_(spikes * 100.0).clamp_(min=0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """報酬に基づく安定的な配線組み換え"""
        with torch.no_grad():
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正: 報酬を tanh で [-1, 1] に正規化し、極端な破壊を防ぐ
            norm_reward = torch.tanh(torch.tensor(reward)).item()
            
            # トレースに基づく状態更新
            self.states.add_(trace * norm_reward * 2.0)
            
            # 自然な忘却
            self.states.sub_(0.02)
            
            # 修正: 強力な「ランダム・スプラウト(発芽)」
            # 結合率が低い時、死んでいるシナプスを強制的に復活させる
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate < 0.05:
                revive_prob = 0.01 + (0.05 - conn_rate)
                revive_mask = torch.rand_like(self.states) < revive_prob
                self.states[revive_mask] = float(self.threshold + 5)

            self.states.clamp_(1, self.max_states)
