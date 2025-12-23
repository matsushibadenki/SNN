# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (生命力維持報酬版)

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
                
                # スパイクが一致すれば +1.0, 余計なスパイクは -0.2
                match = torch.sum(t_f * o_f)
                excess = torch.sum((1-t_f) * o_f)
                reward = float(match.item() - 0.2 * excess.item())
                
                # ネットワークが沈黙している(outが全0)場合、学習を促すための小さな負の報酬
                if out.sum() == 0:
                    reward = -0.1
            
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=reward)
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=reward)
            
            surprise = 0.0
            if self.deep_process.last_error is not None:
                # Surpriseを正規化して監視
                surprise = float(self.deep_process.last_error.abs().mean().item())
            
        return {
            "prediction_error": surprise,
            "reward": reward,
            "output_spike_count": float(out.sum().item())
        }
