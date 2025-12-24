# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 空間局所性強化版・1.58ビットロジックゲートレイヤー
# 目的: 発火ニューロン限定の可塑性を導入し、接続の飽和を防ぎ精度を90%へ引き上げる。

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
        
        # 初期状態: ややスパース（30%程度）に設定
        states = torch.randn(out_features, in_features) * 3.0 + (self.threshold - 5)
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
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # エリジビリティ・トレースをスパイク強度と直結
            self.eligibility_trace.mul_(0.7).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: 特定ニューロンの独占を防ぐ
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(1.0, 15.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 空間局所性を考慮した成功報酬型学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = self.proficiency.item()
            
            # 発火したニューロンのインデックスを取得 (空間局所性の担保)
            # post_spikesがNoneの場合、出力層からの情報を期待
            fired_mask = (post_spikes > 0).float().view(-1, 1) if post_spikes is not None else 1.0
            
            lr = 8.0 * (1.0 - prof * 0.4)
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.clamp(torch.tensor(reward), -1.0, 1.0).item()
            
            if modulation > 0:
                # 成功時: 「発火して貢献した」接続のみを強化
                update = trace * modulation * lr * fired_mask
                self.states.add_(update)
                # 発火しなかったニューロンの接続は微減させてスパース性を維持
                self.states.sub_(0.02 * (1.0 - fired_mask))
            else:
                # 失敗時: 「発火して間違わせた」接続をリセット
                self.states.sub_(trace * abs(modulation) * lr * 0.5 * fired_mask)
            
            # 接続密度の厳格制御
            conn_rate = float(self.get_ternary_weights().mean().item())
            target_conn = 0.25 # ロジックを組むのに最適な密度
            
            if conn_rate > target_conn:
                # 密度超過時は、ランダムに全体を押し下げる（競合の発生）
                self.states.sub_(1.0)
            elif conn_rate < target_conn - 0.05:
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
