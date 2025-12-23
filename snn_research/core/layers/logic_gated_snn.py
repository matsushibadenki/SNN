# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (恒常性維持版)
# 修正内容: 発火頻度に基づく適応的閾値を導入し、学習の停滞を解消する。

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
        
        # 内部状態バッファ
        self.register_buffer('synapse_states', torch.randint(
            self.threshold + 5, self.threshold + 15, (out_features, in_features)
        ).float())
        
        # 膜電位バッファ
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        
        # 恒常性(Homeostasis)のための閾値バッファ
        # 各ニューロンが個別の閾値を持ち、発火しすぎると上がり、発火しないと下がる
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.target_firing_rate = 0.05 # 目標発火率

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
        
        # 入力電流
        current = torch.matmul(x, w.t())
        
        # 膜電位更新
        new_v = self.v_mem + current.view(-1)
        
        # 適応的閾値による発火判定
        spikes = (new_v >= self.v_th).to(torch.float32)
        
        # リセットとリーク
        reset_mask = 1.0 - spikes
        updated_v = new_v * reset_mask * 0.8
        self.v_mem.copy_(updated_v)
        
        # --- 恒常性の更新 ---
        with torch.no_grad():
            # 発火したら閾値を上げ、発火しなければ下げる (時定数は緩やか)
            self.v_th.add_( (spikes - self.target_firing_rate) * 0.05 )
            self.v_th.clamp_(0.5, 10.0) # 極端な値を抑制
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """可塑性の強さを動的に調整"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # LTP: 報酬期待値がない場合でも、相関があれば微増させる
            self.states.add_(correlation * 0.5)
            
            # LTD: ミスマッチの抑制
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= 0.1
            
            # 構造的な減衰 (使われない配線は自然に消える)
            self.states.sub_(0.01)
            
            self.states.clamp_(1, self.max_states)
