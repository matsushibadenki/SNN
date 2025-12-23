# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 精密制御版・1.58ビット・ロジックゲート樹状突起レイヤー
# 目的: シナプス競合と動的学習率を導入し、Acc 90%超を目指す。

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
        
        # 初期状態: 閾値付近でスパース性を維持する分布
        states = torch.randn(out_features, in_features) * 5.0 + (self.threshold - 10)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 0 または 1 のバイナリ（ロジックゲート）表現
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力スパイクと重みの論理積和
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # リーキー積分ダイナミクス (時定数の最適化)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # エリジビリティ・トレース: 発火したニューロンと入力の相関を記録
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
            
            # ホメオスタシス: 過剰発火を抑制し、活動を分散させる
            self.adaptive_threshold.add_((spikes - 0.1) * 0.05)
            self.adaptive_threshold.clamp_(2.0, 15.0)
        
        # リセット: 発火後は電位を下げる
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 習熟度とシナプス競合を考慮した強化型学習則 """
        with torch.no_grad():
            # 報酬の判定（シミュレーション側の報酬設計に合わせる）
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            
            # 習熟度が高いほど学習率を下げる（安定化）
            learning_gain = 2.0 / (1.0 + self.proficiency.item() * 5.0)
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            # 報酬の正規化
            modulation = torch.clamp(torch.tensor(reward), -1.0, 1.0).item()
            
            if modulation > 0:
                # 成功時: 寄与したパスを強化
                self.states.add_(trace * modulation * learning_gain * 5.0)
                # シナプス競合: 全体が1に寄らないよう、非寄与パスを微減
                self.states.sub_(0.01 * self.max_states * (1.0 - trace / 3.0))
            else:
                # 失敗時: 寄与したパスを弱体化
                self.states.sub_(trace * abs(modulation) * learning_gain * 3.0)
            
            # 接続密度の自動調整 (Conn: 10%～40%を目標にする)
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate < 0.1:
                self.states.add_(0.5)
            elif conn_rate > 0.4:
                self.states.sub_(0.5)

            self.states.clamp_(1, self.max_states)
