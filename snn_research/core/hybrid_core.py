# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: Leaky Top-K)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class LeakyTopKActivation(nn.Module):
    """
    Leaky Top-K Activation:
    上位k%はそのまま通し、それ以外も完全にゼロにはせず、
    leak_factor倍して通す。ReLUに対するLeaky ReLUのような関係。
    """
    def __init__(self, sparsity: float = 0.15, gain: float = 3.0, leak_factor: float = 0.1) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.gain = gain
        self.leak_factor = leak_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[1] * self.sparsity)
        if k < 1: k = 1
        
        # 上位k個のインデックスを取得
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        # マスク作成: 上位は1.0, 下位はleak_factor
        mask = torch.full_like(x, self.leak_factor)
        mask.scatter_(1, topk_indices, 1.0)
        
        # 信号増幅
        return x * mask * self.gain

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # 【修正】Leaky Top-K を採用
        # sparsity 0.15: エリート選別
        # leak 0.1: 下位ニューロンも10%の影響力を持つ（文脈保持）
        self.activation = LeakyTopKActivation(sparsity=0.15, gain=3.0, leak_factor=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.norm(x)
        return self.activation(x)

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features, mode='reservoir')
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features, mode='readout')

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None, learning_rate: float = 0.02) -> Dict[str, float]:
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
