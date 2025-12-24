# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (精度向上・安定化版)
# 目的: 接続率の暴走を抑制し、学習ゲインの自動調整によって獲得した知識の破壊を防ぐ。

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
        
        # 初期状態: 閾値付近でより繊細な初期分布（標準偏差を抑える）
        states = torch.randn(out_features, in_features) * 2.0 + (self.threshold - 2.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 0/1のバイナリ接続（1.58ビットロジックの核）
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 接続数に依存しない正規化された電流入力 (接続数でのスケーリングを追加)
        conn_sum = w.sum(dim=1).clamp(min=1.0)
        current = torch.matmul(x, w.t()).view(-1) / torch.sqrt(conn_sum) * 5.0
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 積分定数を 0.9 に上げ、時間的な情報をより保持
        v_mem.mul_(0.9).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 減衰を少し緩やかに (0.9 -> 0.95)
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
            
            # 発火率ホメオスタシス: 目標発火率を少し下げてスパース性を維持
            self.adaptive_threshold.add_((spikes - 0.05) * 0.05)
            self.adaptive_threshold.clamp_(1.0, 20.0)
        
        # ソフトリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 習熟度に基づき学習率を動的に減衰させ、密度の急変を抑える改良学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            # 報酬の非線形変換をマイルドに
            modulation = torch.tanh(torch.tensor(reward / 2.0)).item()
            
            # 学習率を大幅に抑制 (15.0 -> 2.0) し、精度が高まるほど慎重に更新
            base_lr = 2.0 * (1.0 - prof * 0.8)
            
            if modulation > 0:
                # 成功時はトレースに従い強化
                self.states.add_(trace * modulation * base_lr)
            else:
                # 失敗時は弱めるが、構造を壊しすぎないよう係数を小さく
                self.states.sub_(trace * abs(modulation) * base_lr * 0.2)
            
            # 弾性的密度制御の平滑化: 急激な加減算を止め、差分比例に変更
            conn_rate = float(self.get_ternary_weights().mean().item())
            target_min, target_max = 0.15, 0.35
            
            if conn_rate > target_max:
                # 密度超過時は、超過量に応じて穏やかに減衰
                self.states.sub_((conn_rate - target_max) * 0.5)
            elif conn_rate < target_min:
                # 密度不足時
                self.states.add_((target_min - conn_rate) * 0.2)

            self.states.clamp_(1, self.max_states)
