# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: 型安全性向上版)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        
        # 隠れ層: 'reservoir'
        self.fast_process = LogicGatedSNN(in_features, hidden_features, mode='reservoir')
        
        # パススルー
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        # 出力層: 'readout'
        self.output_gate = LogicGatedSNN(hidden_features, out_features, mode='readout')

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
            
            loss_val = 0.0
            acc = 0.0
            
            if target is not None:
                # 2. Compute Error
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                error = target_onehot - out
                
                # 3. Update Weights
                self.output_gate.update_plasticity(r, out, reward=error)

                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                loss_val = error.pow(2).mean().item()

            # --- 統計情報の収集 ---
            # Mypyエラー回避のため、torch関数を使用
            res_density = torch.mean(f).item()
            out_density = torch.mean(out).item()
            
            # membrane_potentialをTensorとしてキャスト
            v_mem = cast(torch.Tensor, self.output_gate.membrane_potential)
            v_mean = torch.mean(v_mem).item()
            v_max = torch.max(v_mem).item()

        return {
            "loss": loss_val,
            "accuracy": acc,
            "res_density": res_density,
            "out_density": out_density,
            "out_v_mean": v_mean,
            "out_v_max": v_max
        }
