# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (代謝・自己組織化版)
# 目的: デッドロックを打破するため、膜電位の揺らぎと構造の代謝（スクラップ＆ビルド）を導入する。

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
        
        # 初期状態: 閾値をまたぐような広い分布で、多様な初期接続を確保
        states = torch.randn(out_features, in_features) * 10.0 + (self.threshold)
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
        
        # 適応的入力ゲイン: 接続数が少ない場合の感度を大幅に上げる
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 接続が疎であるほど、個々のスパイクの価値を高める (1/sqrt -> 1/pow(0.3))
        gain = 15.0 / conn_count.pow(0.3)
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        
        # 膜電位の自律的揺らぎ (確率的発火の促進)
        if self.training:
            # 活動停止を防ぐための微弱なベースライン電流
            current.add_(torch.randn_like(v_mem) * 0.2 + 0.1)
        
        v_mem.mul_(0.9).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 減衰をやや緩やかにし、因果関係を保持
            self.eligibility_trace.mul_(0.92).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: 目標発火率 5%
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(2.0, 20.0)
        
        # ソフトリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 構造代謝学習則: 役に立たない接続を積極的に捨て、新しい接続を試す """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率の調整
            lr = 2.5 * (1.0 - prof * 0.5)
            
            # 1. 報酬に基づく更新
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時は、その活動に寄与したシナプスを大幅に弱体化（代謝の促進）
                self.states.sub_(trace * abs(modulation) * lr * 1.5)
            
            # 2. 自然減衰とランダム・プロモーション（代謝システム）
            # 接続されていないシナプス（states < threshold）を稀にランダムに上昇させる
            noise = torch.randn_like(self.states) * 0.05
            self.states.add_(noise)
            
            # 3. 厳格な密度ホメオスタシス (ターゲット 15-25%)
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate > 0.25:
                self.states.sub_(0.5) # 密度超過時は一斉に冷やす
            elif conn_rate < 0.15:
                self.states.add_(0.2) # 密度不足時は一斉に温める

            self.states.clamp_(1, self.max_states)
