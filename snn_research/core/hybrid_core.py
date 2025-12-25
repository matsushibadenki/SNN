# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Final Fix: 差分ゲーティング)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class DiffGatedTopK(nn.Module):
    """
    Difference-based Gating Top-K:
    上位1位と2位の差分（Confidence Margin）に基づいて、信号強度を動的に調整する。
    差が大きい＝確信度が高い -> 強く通す
    差が小さい＝迷っている -> 弱く通す（抑制）
    """
    def __init__(self, sparsity: float = 0.15, gain: float = 3.0) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Top-K マスク
        k = int(x.shape[1] * self.sparsity)
        if k < 2: k = 2 # 差分計算のため最低2つ必要
        
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        # 1位と2位の差分を計算 (バッチごと)
        # topk_values: [Batch, K]
        diff = topk_values[:, 0] - topk_values[:, 1] # [Batch]
        
        # 差分に応じた動的ゲイン (Sigmoidでスケーリング)
        # 差が0に近い -> ゲイン小
        # 差が大きい -> ゲイン大
        dynamic_gain = torch.sigmoid(diff) * self.gain + 1.0
        dynamic_gain = dynamic_gain.unsqueeze(1) # [Batch, 1]
        
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_indices, 1.0)
        
        # マスク * 動的ゲイン
        return x * mask * dynamic_gain

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # 【修正】Diff-Gated Top-K を採用
        self.activation = DiffGatedTopK(sparsity=0.15, gain=3.0)
        
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
