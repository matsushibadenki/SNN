# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (探索促進報酬版)

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
                
                # スパイク一致でプラス、不一致または「沈黙」にペナルティ
                hits = torch.sum(t_f * o_f)
                if hits > 0:
                    reward = float(hits.item() * 5.0)
                elif out.sum() == 0:
                    reward = -1.0 # 沈黙に対する明確な罰
                else:
                    reward = -0.5 # 誤発火に対する罰
            
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=reward)
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=reward)
            
            surprise = 0.0
            if self.deep_process.last_error is not None:
                surprise = float(self.deep_process.last_error.pow(2).mean().item())
            
        return {
            "prediction_error": surprise,
            "reward": reward,
            "output_spike_count": float(out.sum().item())
        }
