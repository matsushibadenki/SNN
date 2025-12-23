# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (動的再配線・活性化版)
# 目的: 安定性を保ちつつ結合密度を10-15%へ引き上げ、情報の表現容量を拡大して精度を爆発させる。

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
        
        # 初期状態: 閾値付近に設定
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold - 1.0)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正1: 目標結合密度を15%に設定
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
            # 修正2: トレースの累積を強め、因果関係をはっきりさせる
            self.eligibility_trace.mul_(0.7).add_(torch.outer(spikes, x.view(-1)) * 3.0)
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火の平準化 (ホメオスタシス)
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 競争的成長と環境適応型プルーニング """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正3: 報酬利得の強化
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            if modulation > 0:
                # 成功時: 強く固定
                self.states.add_(trace * modulation * 15.0)
            else:
                # 失敗時: 関連配線を鋭く削除
                self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正4: 構造的可塑性の動的調整 (15%を維持する力)
            if conn_rate < self.target_conn_rate:
                # 密度が足りない場合は、全配線をわずかに底上げ（浮上を待つ）
                self.states.add_(0.5)
            else:
                # 密度過多の場合は、ランダム性を伴うプルーニング
                self.states.sub_(0.2)
                # 活動のない配線を時々リセットして探索を促す
                inactive_mask = (torch.rand_like(self.states) < 0.001) & (self.states <= self.threshold)
                self.states[inactive_mask] = float(self.threshold - 5)

            self.states.clamp_(1, self.max_states)
