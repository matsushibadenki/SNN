# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 進化的剪定・高精度ロジックゲートレイヤー
# 目的: ソフト空間局所性と進化的剪定を導入し、認識精度 90% 以上を確実に達成する。

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
        
        # 初期状態: 広めに分布させて探索を促進
        states = torch.randn(out_features, in_features) * 10.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
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
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 情報を蓄積しやすくするために保持率を0.9に
        v_mem.mul_(0.9).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # エリジビリティ・トレース: 入出力の相関をより長く保持
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: 発火率を一定に保つ
            self.adaptive_threshold.add_((spikes - 0.1) * 0.05)
            self.adaptive_threshold.clamp_(2.0, 20.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = self.proficiency.item()
            
            # ソフト空間局所性: 発火したニューロンを1.0、しなかったニューロンを0.1の重みで更新
            # これにより、未発火ニューロンもわずかに学習機会を得る
            fired_mask = (post_spikes > 0).float().view(-1, 1) if post_spikes is not None else torch.ones(self.out_features, 1)
            soft_mask = fired_mask + 0.1 * (1.0 - fired_mask)
            
            # 学習率の動的調整
            lr = 15.0 / (1.0 + prof * 5.0)
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.clamp(torch.tensor(reward), -1.0, 1.0).item()
            
            if modulation > 0:
                # 成功時: 寄与した接続を強化 (Soft Mask適用)
                self.states.add_(trace * modulation * lr * soft_mask)
            else:
                # 失敗時: 寄与した接続を弱体化
                self.states.sub_(trace * abs(modulation) * lr * 0.5 * soft_mask)
            
            # 進化的剪定 (Evolutionary Pruning)
            # 接続率が高い場合は、弱い接続から順に削る
            conn_rate = float(self.get_ternary_weights().mean().item())
            target_conn = 0.35 - (prof * 0.15) # 習熟に従い35%から20%へ
            
            if conn_rate > target_conn:
                # 接続が多い場合、トレース（貢献度）が低いシナプスを優先的に減衰させる
                decay = (1.0 - trace / 5.0) * 0.5
                self.states.sub_(decay)
            elif conn_rate < target_conn - 0.1:
                # 接続が少なすぎる場合は全体を底上げ
                self.states.add_(0.2)

            self.states.clamp_(1, self.max_states)
