# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (構造固定・論理精錬版)
# 目的: 接続率の暴走を物理的に遮断し、限定されたリソース内での論理最適化を強制する。

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
        
        # 初期状態: ターゲット密度 20% 付近に厳密に配置
        states = torch.randn(out_features, in_features) * 2.0 + (self.threshold - 10.0)
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
        
        # 動的入力スケーリング: 接続数（エネルギー消費）に応じた厳格な抑制
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 接続が多いニューロンほど、一つ一つの信号の重みを小さくする
        current = torch.matmul(x, w.t()).view(-1) / (conn_count / 10.0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # リークを少し早め、スパイクの「キレ」を良くする
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 減衰を早め、直近の因果関係に集中
            self.eligibility_trace.mul_(0.85).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
            
            # ホメオスタシス: 10%の発火率をターゲットにする
            self.adaptive_threshold.add_((spikes - 0.1) * 0.5)
            self.adaptive_threshold.clamp_(4.0, 30.0)
        
        # ハードリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.0)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ ハード・キャッピング学習則: 接続数の上限を物理的に強制する """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            # 報酬の圧縮: 過剰な強化を防ぐ
            modulation = torch.tanh(torch.tensor(reward / 2.0)).item()
            
            # 学習率を安定方向に調整 (5.0 -> 1.2)
            lr = 1.2 * (1.0 - prof * 0.5)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時の忘却を強化（接続の入れ替えを促進）
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # 物理的密度制御 (Global Competition)
            # 全ニューロンの平均接続率を 20% に引き戻す強力な圧力
            conn_rate = float(self.get_ternary_weights().mean().item())
            target_rate = 0.20
            
            # 逸脱量に対して非線形な引き戻しを適用
            pull_force = (conn_rate - target_rate) * 5.0
            self.states.sub_(pull_force)

            # 追加のハードキャップ: 接続率が40%を超えたら強制デクリメント
            conn_per_neuron = (self.states > self.threshold).float().mean(dim=1)
            self.states.sub_((conn_per_neuron > 0.4).float().unsqueeze(1) * 2.0)

            self.states.clamp_(1, self.max_states)
