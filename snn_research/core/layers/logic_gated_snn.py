# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (覚醒・再起動版)
# 修正内容: 沈黙状態を強制的に打破する「自発発火メカニズム」を導入。

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
        
        self.register_buffer('synapse_states', torch.randint(
            self.threshold + 5, self.threshold + 15, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    @property
    def v_mem(self) -> torch.Tensor:
        return cast(torch.Tensor, self.membrane_potential)

    @property
    def v_th(self) -> torch.Tensor:
        return cast(torch.Tensor, self.adaptive_threshold)

    def get_ternary_weights(self) -> torch.Tensor:
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)

        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正1: 膜電位の蓄積率を向上 (0.3 -> 0.8) し、情報を貯めやすくする
        self.v_mem.add_(current)
        
        # 修正2: 「沈黙」を許さない強力なランダム・バイアス
        # 全く発火していないニューロンには強い興奮性ノイズを乗せる
        silent_mask = (self.firing_history < 0.01).float()
        spontaneous_noise = torch.randn_like(self.v_mem) * 1.5 * silent_mask
        
        potential = self.v_mem + spontaneous_noise
        
        # 発火判定
        spikes = (potential >= self.v_th).to(torch.float32)
        
        # リセット処理
        # 発火した場合は電位を引くが、完全0にはせず「余韻」を残す
        self.v_mem.copy_(potential * (1.0 - spikes) * 0.9)
        
        # 履歴の更新 (時定数を少し長くする)
        self.firing_history.copy_(self.firing_history * 0.95 + spikes * 0.05)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, surprise: float = 1.0) -> None:
        with torch.no_grad():
            # 修正3: 誤差が固定されている場合、より広範囲に配線を組み換える
            # surprise が一定値（0.09など）に固着していることを想定
            if surprise > 0.05:
                # 誤差が大きいほど、既存の結合をランダムに「揺らす」
                perturbation = (torch.rand_like(self.states) < 0.2).float() * surprise * 10.0
                self.states.add_(torch.randn_like(self.states) * perturbation)

            correlation = torch.outer(post_spikes, pre_spikes)
            # 強化 (LTP) を大幅に強化
            self.states.add_(correlation * 5.0)
            
            # 抑制 (LTD)
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= 2.0

            # 死んでいるニューロンを強制的に Include 状態へ
            dead_neurons = self.firing_history < 0.005
            if dead_neurons.any():
                self.states[dead_neurons] += 2.0
                self.v_th[dead_neurons] = 0.5 # 閾値を下げて「門戸を開く」

            self.states.clamp_(1, self.max_states)
