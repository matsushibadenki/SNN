# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (長期記憶・安定化版)
# 目的: シナプス接続の急激な消失を防ぎ、獲得した論理構造を保護・固定する。

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
        
        # 初期状態: 接続率を15%程度に抑えた初期分布
        states = torch.randn(out_features, in_features) * 3.0 + (self.threshold - 8.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # バイナリ重み: 接続の有無を決定
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 接続数による過剰入力を防ぐ正規化 (Softmax的な感度調整)
        conn_sum = w.sum(dim=1).clamp(min=1.0)
        current = torch.matmul(x, w.t()).view(-1) / torch.log1p(conn_sum) * 2.0
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 膜電位のリークを遅くし、情報を蓄積 (0.9 -> 0.95)
        v_mem.mul_(0.95).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: より長期の依存関係を捉える (0.95 -> 0.98)
            self.eligibility_trace.mul_(0.98).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
            
            # 発火率ホメオスタシス: 極端なスパイクを抑制
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(2.0, 25.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 非対称な学習則: 成功の強化を優先し、失敗による切断を慎重に行う """
        with torch.no_grad():
            # 習熟度の更新をより安定させる (時定数を長く)
            is_success = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.999 + is_success * 0.001)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 習熟度が上がるほど学習を「凍結」し、既存の知識を保護
            lr = 1.5 * (1.0 - prof * 0.9)
            
            if modulation > 0:
                # 成功時は積極的に強化
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時の弱体化は非常に慎重に (強化の1/10の速度)
                self.states.sub_(trace * abs(modulation) * lr * 0.1)
            
            # 弾性的密度制御の改良: 習熟度が低い間だけ働き、高まると固定に移行
            conn_rate = float(self.get_ternary_weights().mean().item())
            # 目標密度範囲 10% - 25%
            density_pull = (1.0 - prof) * 0.05
            
            if conn_rate > 0.25:
                # 超過時は、接続が弱いものから削る
                self.states.sub_(density_pull)
            elif conn_rate < 0.10:
                # 不足時は、全体を底上げ
                self.states.add_(density_pull)

            self.states.clamp_(1, self.max_states)
