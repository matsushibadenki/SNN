# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 動的リソース競合型・1.58ビットロジックゲートレイヤー
# 目的: 構文エラーの修正、およびシナプス総和制限による90%超の精度安定化。

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
        
        # 初期状態: 20%程度のスパース性を持つ分布
        states = torch.normal(self.threshold - 4.0, 3.0, (out_features, in_features))
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0))
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
        
        # 接続密度に応じた入力正規化（暴走防止）
        conn_sum = w.sum(dim=1).clamp(min=1.0)
        current = torch.matmul(x, w.t()).view(-1) / torch.sqrt(conn_sum)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(2.0, 15.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 競争的リソース配分による学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            lr = 12.0 * (1.0 - prof * 0.5)
            
            # --- 弾性的・競争的更新 ---
            if modulation > 0:
                # 成功時: 寄与したパスを強化
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時: 全体の接続ポテンシャルを削り、再編を促す
                self.states.sub_(trace * abs(modulation) * lr * 0.4)
            
            # 構造的制約: 各ニューロンの有効接続率を約20%に制限
            w = self.get_ternary_weights()
            conn_counts = w.sum(dim=1)
            target_conn = self.in_features * 0.20
            
            for i in range(self.out_features):
                if conn_counts[i] > target_conn:
                    # 接続過剰な場合、貢献度の低い接続のポテンシャルを削る
                    excess = int(conn_counts[i] - target_conn)
                    active_synapses = (w[i] > 0)
                    _, indices = torch.sort(trace[i] * active_synapses.float(), descending=False)
                    self.states[i, indices[:excess]] -= 2.0
                elif conn_counts[i] < target_conn * 0.5:
                    # 接続不足な場合、トレースがある箇所を底上げ
                    self.states[i].add_(0.1)

            self.states.clamp_(1, self.max_states)
