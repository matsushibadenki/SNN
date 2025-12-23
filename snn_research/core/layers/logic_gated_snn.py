# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (覚醒・改善版)
# 修正内容: 学習の停滞を防ぐため、初期結合率と可塑性の更新強度を最適化。

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
        
        # 修正1: 初期状態を閾値より高めに設定 (初期結合率を上げる)
        # これにより、最初は信号が通りやすく、学習のきっかけを掴みやすくする
        self.register_buffer('synapse_states', torch.randint(
            self.threshold + 5, self.threshold + 15, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 修正2: 膜電位の閾値を調整 (入力の総和に対して適切な感度にする)
        self.spike_threshold = 2.0 

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    @property
    def v_mem(self) -> torch.Tensor:
        return cast(torch.Tensor, self.membrane_potential)

    def get_ternary_weights(self) -> torch.Tensor:
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 修正3: 入力電流の強さを調整
        current = torch.matmul(x, w.t())
        
        new_v = self.v_mem + current.view(-1)
        spikes = (new_v >= self.spike_threshold).to(torch.float32)
        
        # 指数的な減衰を導入 (完全リセットではなくリーキーな特性を微追加)
        reset_mask = 1.0 - spikes
        updated_v = new_v * reset_mask * 0.9 
        self.v_mem.copy_(updated_v)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """可塑性更新を加速"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 修正4: 更新強度を 0.5 -> 2.0 へ。学習の初動を速める。
            self.states.add_(correlation * 2.0)
            
            # 抑圧も少し強める
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= 1.0
            
            self.states.clamp_(1, self.max_states)
