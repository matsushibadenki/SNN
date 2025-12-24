# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (適応的生命維持版)
# 目的: 活動停止（デッドロック）を検知し、閾値の動的調整によって学習能力を強制復元する。

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
        
        # 初期状態: 分散を大きくし、ニューロンごとの個性を出す
        states = torch.randn(out_features, in_features) * 8.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 6.0))
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
        
        # 入力ゲインの動的補償: 接続数が少ない時（飢餓状態）ほどゲインを上げる
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # ログの 0.5% という極端な疎状態でも発火できるよう、補正を強化
        gain = 25.0 / torch.sqrt(conn_count / 10.0 + 0.1)
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 【重要】活動依存のホメオスタシス
            # 発火したら閾値を上げ、発火しない（飢餓）なら急速に下げる
            # ログの V_th=20.0, V_avg=1.1 という絶望的なギャップを埋める
            self.adaptive_threshold.add_((spikes - 0.1) * 0.5)
            # 自然減衰（常に感度を高めようとする圧力）を追加
            self.adaptive_threshold.mul_(0.99)
            self.adaptive_threshold.clamp_(1.5, 30.0)
        
        # ソフトリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 非線形・個別密度制御型学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率の安定化
            lr = 2.0 * (1.0 - prof * 0.5)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時の切断は、現在の接続率が高い時ほど厳しくする
                conn_rate = float(self.get_ternary_weights().mean().item())
                penalty_scale = 1.5 if conn_rate > 0.3 else 0.2
                self.states.sub_(trace * abs(modulation) * lr * penalty_scale)
            
            # 代謝システム（ランダムな揺らぎ）: デッドロックを物理的に防ぐ
            # 1% の確率で非常に弱いノイズを加え、休眠中のシナプスを活性化
            noise = torch.randn_like(self.states) * 0.02
            self.states.add_(noise)
            
            # 密度ホメオスタシス（個別・非線形）
            # 平均ではなく、各ニューロンの活動ポテンシャルを調整
            conn_rates = (self.states > self.threshold).float().mean(dim=1)
            # 接続率 10% - 30% を維持するように、ニューロンごとにフィードバック
            pull = (conn_rates - 0.20) * 2.0
            self.states.sub_(pull.unsqueeze(1))

            self.states.clamp_(1, self.max_states)
