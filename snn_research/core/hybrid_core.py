# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Final Fix: 厳格Top-K & ハイパーゲイン)
# 修正: reset_state メソッドの追加

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class TopKActivation(nn.Module):
    def __init__(self, sparsity: float = 0.10, gain: float = 4.0) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[1] * self.sparsity)
        if k < 1: k = 1
        
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_indices, 1.0)
        
        return x * mask * self.gain

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # sparsity 0.10: 上位10%の「エリート」のみを通す
        # gain 4.0: 選ばれた信号を超強力にブーストする
        self.activation = TopKActivation(sparsity=0.10, gain=4.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.norm(x)
        return self.activation(x)

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features, mode='reservoir')
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features, mode='readout')

    def reset_state(self):
        """全層の状態リセット"""
        self.fast_process.reset_state()
        self.output_gate.reset_state()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None, learning_rate: float = 0.05) -> Dict[str, float]:
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            loss_val = 0.0
            acc = 0.0
            
            if target is not None:
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                error = (target_onehot - out)
                
                self.output_gate.update_plasticity(r, out, reward=error, learning_rate=learning_rate)

                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                loss_val = error.pow(2).mean().item()

            res_density = torch.mean(f).item()
            out_density = torch.mean(out).item()
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
