# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (情報の定着・粘性版)
# 目的: 急激なプルーニングによる情報の消失を防ぎ、Acc 10% 以上の回路を「骨組み」として維持する。

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
        
        # 修正1: 閾値周辺に固め、わずかな報酬で Include/Exclude が切り替わる「感受性」を確保
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 1)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正2: 認識に必要な最小限の「情報の太さ」を確保 (15% - 25%)
        self.target_conn_rate = 0.20

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 修正3: 非線形増幅の強化 (特定の経路が一致した時だけ爆発的に電位を上げる)
        raw_current = torch.matmul(x, w.t()).view(-1)
        current = torch.pow(raw_current, 1.5) # 重畳の一致を強調
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.7).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新 (活動の記憶)
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: 発火率の適正化
            self.adaptive_threshold.add_((spikes - 0.1) * 0.05)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 粘性を持たせた『彫刻』学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正4: 更新の歩幅を小さくし、情報の「蒸発」を防ぐ
            if modulation > 0:
                # 成功時: 確実な定着
                self.states.add_(trace * modulation * 5.0)
            else:
                # 失敗時: 痕跡箇所を削るが、一気にゼロにはしない
                self.states.sub_(trace * abs(modulation) * 2.0)
            
            # 修正5: 構造的リザーバ (密度の安定化)
            # 20% を目指して「極めてゆっくり」と調整する
            if conn_rate > self.target_conn_rate:
                # 密度過多の場合のみ、微弱な剪定圧をかける
                self.states.sub_(0.02)
            elif conn_rate < 0.10:
                # 密度が低すぎる場合は、ランダムに小規模な「発芽」を促す
                revive_mask = torch.rand_like(self.states) < 0.005
                self.states[revive_mask] += 5.0

            self.states.clamp_(1, self.max_states)
