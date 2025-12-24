# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: 単純化デルタ則版)

import torch
import torch.nn as nn
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features): super().__init__()
    def forward(self, x): return x

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        # 隠れ層
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        
        # パススルー
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        # 出力層
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        # Feedback Alignment Matrix
        # 出力層の誤差を隠れ層に伝えるための固定重み
        self.register_buffer(
            'feedback_matrix', 
            torch.randn(out_features, hidden_features)
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.no_grad():
            # 1. Forward Pass
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar_val = 0.0
            
            if target is not None:
                # 2. Compute Error (Target - Output)
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                # 単純明快な誤差信号
                # Target=1, Out=0 -> Error=+1 (結合を強めろ)
                # Target=0, Out=1 -> Error=-1 (結合を弱めろ)
                error = target_onehot - out
                
                # 3. Update Output Layer
                self.output_gate.update_plasticity(r, out, reward=error)
                
                # 4. Update Hidden Layer (via Feedback Alignment)
                # 出力層での誤差を隠れ層に投影
                hidden_error = torch.matmul(error, self.feedback_matrix)
                
                # 隠れ層も同じロジックで成長させる
                self.fast_process.update_plasticity(x_input, f, reward=hidden_error)

                # 精度計測
                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                reward_scalar_val = acc

        return {
            "prediction_error": 0.0,
            "reward": reward_scalar_val,
            "output_spike_count": float(out.sum().item() / x_input.size(0)),
            "proficiency": float(self.output_gate.proficiency.item())
        }
