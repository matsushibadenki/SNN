# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (リバース・アニーリング版)
# 目的: 学習初期の過剰なプルーニングを抑制し、高精度(80%超)を達成した後に洗練フェーズへ移行する。

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
        
        # 修正1: 閾値直上に配置。最初はすべての配線が「生きている」状態でスタート
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold + 2)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 知識の習熟度 (0.0: 未熟 -> 1.0: 熟練)
        self.register_buffer('proficiency', torch.zeros(1))
        self.target_conn_rate = 0.15 

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
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 習熟度に応じて『彫刻の鋭さ』を変える学習則 """
        with torch.no_grad():
            # 成功時に習熟度を上げ、失敗時に少し下げる
            prof_delta = 0.01 if reward > 1.0 else -0.005
            self.proficiency.add_(prof_delta).clamp_(0.0, 1.0)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正2: 習熟度が低い(0.3以下)間はプルーニング(削り)を物理的に禁止する
            # これにより、初期の 80% 密度を維持して確実に知識を捕獲する
            pruning_enable = 1.0 if self.proficiency.item() > 0.3 else 0.0
            
            if modulation > 0:
                # 成功時: 強く固定
                self.states.add_(trace * modulation * 15.0)
            else:
                # 失敗時: プルーニングが有効なら削る
                self.states.sub_(trace * abs(modulation) * 10.0 * pruning_enable)
            
            # 修正3: 密度調整も習熟度に従う
            if pruning_enable > 0 and conn_rate > self.target_conn_rate:
                # 賢くなった後だけ、贅肉を削ぎ落とす
                self.states.sub_(0.2 * (conn_rate / self.target_conn_rate))
            elif conn_rate < 0.05:
                # 生命維持
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
