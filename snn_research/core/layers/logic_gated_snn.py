# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (疎構造・恒常性維持版)
# 目的: 100%接続という飽和状態を物理的に禁止し、疎な論理ゲートの自己組織化を強制する。

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
        
        # 初期状態: 接続率 5% 程度の非常に疎な状態から開始
        states = torch.randn(out_features, in_features) * 2.0 + (self.threshold - 20.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 10.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力正規化: 接続数が多いほど個々の信号を極端に弱める (負のフィードバック)
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 接続が100%に近づくと入力がほぼゼロになるように設計
        current = torch.matmul(x, w.t()).view(-1) / (conn_count.pow(1.5) / 10.0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 短期的な相関を重視
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
            
            # 強烈なホメオスタシス: 発火したら閾値を爆発的に上げる (15.0までではなく無限遠まで)
            self.adaptive_threshold.add_((spikes - 0.1) * 5.0)
            self.adaptive_threshold.clamp_(5.0, 100.0)
            self.adaptive_threshold.mul_(0.99) # 徐々に回復
        
        # ハードリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.0)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ シナプス崩壊（Decay）を導入した自然淘汰型学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率の適正化
            lr = 2.0 * (1.0 - prof * 0.5)
            
            # 基本減衰 (シナプスのL2正則化): 常に接続を「切る」方向に圧力をかける
            # これにより、報酬が得られない接続は自然に消滅する
            self.states.sub_(0.1)
            
            if modulation > 0:
                # 成功した時のみ、減衰に打ち勝つ強化を与える
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時は、微小なペナルティ
                self.states.sub_(trace * abs(modulation) * 0.1)
            
            # 強制的な密度キャップ: 接続率が40%を超えたら、そのニューロンの全シナプスを冷却
            conn_rates = (self.states > self.threshold).float().mean(dim=1)
            over_saturated = (conn_rates > 0.40).float().unsqueeze(1)
            self.states.sub_(over_saturated * 5.0)

            self.states.clamp_(1, self.max_states)
