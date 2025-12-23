# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (容量確保・認識覚醒版)
# 目的: 結合密度を意図的に 10-15% 程度まで押し上げ、Acc 50% 超を実現するための情報帯域を確保する。

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
        
        # 初期状態: 閾値付近に設定し、最初は「つながりやすい」状態にする
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold + 1.0)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正1: 結合密度の目標を 15% に引き上げ (表現容量の拡大)
        self.target_conn_rate = 0.15

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
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: トレースの「蓄積」を強化し、稀な正解パターンを逃さない
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)) * 4.0)
            self.eligibility_trace.clamp_(0, 8.0)
            
            # 発火の平準化
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 厳格な密度制御を緩め、学習の「勢い」を許容する """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正3: 報酬利得の爆発的強化
            if modulation > 0:
                # 成功時は、現在の密度に関わらず配線を強く固定する
                self.states.add_(trace * modulation * 15.0)
            else:
                # 失敗時の剪定をマイルドにし、せっかく芽生えた知能を守る
                self.states.sub_(trace * abs(modulation) * 3.0)
            
            # 修正4: 目標密度 15% への誘引 (下限を強力に押し上げる)
            if conn_rate < self.target_conn_rate:
                # 密度が足りない時は積極的に「発芽」させる
                growth = (self.target_conn_rate - conn_rate) * 5.0
                self.states.add_(growth)
            elif conn_rate > 0.4: # 40% を超えた時だけ緊急ブレーキ
                self.states.sub_(2.0)
            
            self.states.clamp_(1, self.max_states)
