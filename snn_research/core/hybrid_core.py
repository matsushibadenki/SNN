# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (予測誤差連動学習版)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
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
            # 1. 順伝播
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            # 2. 「驚き (Surprise)」の算出
            # 予測誤差が大きいほど Surprise が高くなる
            surprise = 0.0
            if self.deep_process.last_error is not None:
                surprise = float(self.deep_process.last_error.abs().mean().item())
            
            # 3. 局所学習則への Surprise フィードバック
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), surprise=surprise)
            
            # 4. 出力層の学習 (targetとの差異を Surprise として扱う)
            output_surprise = 0.0
            if target is not None:
                output_surprise = float((target.view(-1) - out).abs().mean().item())
            
            feedback = target.view(-1) if target is not None else r.view(-1)
            self.output_gate.update_plasticity(r.view(-1), feedback, surprise=output_surprise)
            
        return {
            "prediction_error": surprise,
            "fast_layer_states_avg": float(self.fast_process.states.mean().item()),
            "output_spike_count": float(out.sum().item())
        }
