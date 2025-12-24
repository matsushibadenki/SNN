# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: LSM構成・堅牢版)

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
        
        # 隠れ層: 'reservoir' モード (固定・量子化重み)
        self.fast_process = LogicGatedSNN(in_features, hidden_features, mode='reservoir')
        
        # パススルー (将来的な拡張用)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        # 出力層: 'readout' モード (学習・連続値重み)
        self.output_gate = LogicGatedSNN(hidden_features, out_features, mode='readout')

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        自律学習ステップ
        """
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
                
                # 誤差計算: 教師信号 - 出力
                error = target_onehot - out
                
                # 3. Update Output Layer ONLY
                self.output_gate.update_plasticity(r, out, reward=error)

                # 精度計測
                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                reward_scalar_val = acc

        return {
            "prediction_error": 0.0,
            "reward": reward_scalar_val,
            "output_spike_count": float(out.sum().item() / max(1, x_input.size(0))),
            "proficiency": reward_scalar_val
        }
