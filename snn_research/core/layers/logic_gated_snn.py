# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (確率的・拮抗的学習版)
# 修正内容: 確率的発火と側方抑制を導入し、同一パターンへの固着を物理的に排除する。

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
        
        # 内部状態
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 10, self.threshold + 10, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        
        # 側方抑制用の重み (自分以外を弱く叩く)
        self.inhibition_strength = 0.5

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
        
        # 1. 入力電流の計算
        current = torch.matmul(x, w.t()).view(-1)
        
        # 2. 確率的発火判定 (熱ノイズの導入)
        # 膜電位が閾値に近いほど、確率的に発火する (シグモイド近似)
        noise = torch.randn_like(current) * 0.2
        potential_with_noise = self.v_mem + current + noise
        
        spikes = (potential_with_noise >= self.v_th).to(torch.float32)
        
        # 3. 側方抑制 (Lateral Inhibition)
        # 誰かが発火したら、全体の膜電位を下げる (過剰発火の抑制)
        if spikes.any():
            inhibition = spikes.sum() * self.inhibition_strength
            self.v_mem.sub_(inhibition)
        
        # 4. リセットと更新
        reset_mask = 1.0 - spikes
        self.v_mem.copy_( (self.v_mem + current) * reset_mask * 0.5 )
        
        # 5. 恒常性 (発火頻度の平準化)
        with torch.no_grad():
            # 発火したら閾値を上げ、しなければ下げる
            self.v_th.add_( (spikes - 0.1) * 0.01 )
            self.v_th.clamp_(0.2, 5.0)
            
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, surprise: float = 1.0) -> None:
        """予測誤差(驚き)を学習率として使用"""
        with torch.no_grad():
            # 驚きが大きいほど、大幅に配線を変更する
            lr = 1.0 + surprise * 10.0
            
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # LTP: 相関があれば強化
            self.states.add_(correlation * lr)
            
            # LTD: 期待外れの場合の弱体化
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= (0.5 * lr)
            
            # 全体的な忘却 (構造的可塑性の基礎)
            self.states.sub_(0.05)
            self.states.clamp_(1, self.max_states)
