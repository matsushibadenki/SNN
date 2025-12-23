# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (能動探索報酬版)

import torch
import torch.nn as nn
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward = 0.0
            if target is not None:
                t_f = target.view(-1)
                o_f = out.view(-1)
                hits = torch.sum(t_f * o_f)
                misses = torch.sum((1 - t_f) * o_f)
                
                # 修正: 沈黙（全ゼロ）は「最大の罰」
                if out.sum() == 0:
                    reward = -5.0
                else:
                    # わずかな成功でも大きな報酬、失敗は最小限に
                    reward = float(hits.item() * 10.0 - misses.item() * 1.0)
            
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=reward)
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=reward)
            
            surprise = 0.0
            if self.deep_process.last_error is not None:
                surprise = float(self.deep_process.last_error.abs().mean().item())
            
        return {
            "prediction_error": surprise,
            "reward": reward,
            "output_spike_count": float(out.sum().item())
        }
