# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (自己組織化・堅牢版)
# 目的: 入力正規化の厳格化とシナプス状態の飽和制御により、100%接続などの暴走を物理的に不可能にする。

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
        
        # 初期状態: より低い初期密度 (約5-10%) から開始し、慎重に探索
        states = torch.randn(out_features, in_features) * 2.0 + (self.threshold - 15.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 8.0))
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
        
        # 入力正規化の厳格化: 接続数と入力密度の両方でスケーリング
        # 100%接続時でも入力電流が爆発しないように log スケールを強化
        conn_count = w.sum(dim=1).clamp(min=1.0)
        input_density = x.mean().clamp(min=0.01)
        
        raw_current = torch.matmul(x, w.t()).view(-1)
        # 「入力スパイク数 / 接続数」の比率をベースにし、感度を調整
        current = (raw_current / conn_count) * 15.0 / input_density.sqrt()
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 膜電位のリークを早め、過去の過剰な入力を素早く忘却 (0.95 -> 0.85)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 成功への寄与度を鋭く判定
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
            
            # 発火率ホメオスタシス: 活動過多時は閾値を「高速」に上げる
            self.adaptive_threshold.add_((spikes - 0.05) * 1.5) 
            self.adaptive_threshold.clamp_(3.0, 40.0)
            # 自然減衰による閾値の低下
            self.adaptive_threshold.sub_(0.01)
        
        # 強いリセット (Hard Reset)
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.0)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 飽和抑制付き学習則: 状態が閾値に近づくほど更新を鈍らせる """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 2.0)).item()
            
            # 学習率をさらに絞る (2.0 -> 0.8)
            lr = 0.8 * (1.0 - prof * 0.5)
            
            # 状態依存のゲート: 閾値付近での急激なON/OFFを抑制
            # 状態が 45-55 の範囲にある時、更新をより慎重にする
            dist_from_threshold = (self.states - self.threshold).abs()
            soft_gate = torch.clamp(dist_from_threshold / 10.0, 0.1, 1.0)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr * soft_gate)
            else:
                self.states.sub_(trace * abs(modulation) * lr * 0.5 * soft_gate)
            
            # 弾性的密度制御: 逸脱量に対して指数的な引き戻しを適用
            conn_rate = float(self.get_ternary_weights().mean().item())
            target_rate = 0.15 # 15%を維持
            
            # 密度の偏りを補正するための微小な定数圧
            diff = conn_rate - target_rate
            self.states.sub_(diff * 2.0) 

            self.states.clamp_(1, self.max_states)
