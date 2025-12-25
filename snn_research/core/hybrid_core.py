# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: k-WTA & Top-K Filtering)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class TopKActivation(nn.Module):
    """
    k-Winner-Take-All Activation:
    上位k%の強い信号のみを通し、それ以外を抑制する。
    圧倒的なノイズ除去能力を持つ。
    """
    def __init__(self, sparsity: float = 0.1) -> None:
        super().__init__()
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # k個のニューロンを選定
        k = int(x.shape[1] * self.sparsity)
        if k < 1: k = 1
        
        # 上位k個の値とインデックスを取得
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        # 出力テンソルをゼロで初期化（抑制）
        mask = torch.zeros_like(x)
        
        # 上位k個の場所に1を立てる（微分のためにscatterを使用）
        # 値自体はそのまま通す（ReLUのような挙動）
        mask.scatter_(1, topk_indices, 1.0)
        
        return x * mask

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # GELUの代わりにTop-Kを使用。
        # 上位10%の「確信度の高い特徴」だけを次層に送る
        self.activation = TopKActivation(sparsity=0.10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.norm(x)
        return self.activation(x)

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        
        # 隠れ層: 'reservoir'
        self.fast_process = LogicGatedSNN(in_features, hidden_features, mode='reservoir')
        
        # パススルー + 正規化 + k-WTA (堅牢性強化)
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
            r = self.deep_process(f) # ここで強力なデノイズがかかる
            out = self.output_gate(r)
            
            loss_val = 0.0
            acc = 0.0
            
            if target is not None:
                # 2. Compute Error
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                error = target_onehot - out
                
                # 3. Update Weights (with Momentum)
                self.output_gate.update_plasticity(r, out, reward=error)

                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                loss_val = error.pow(2).mean().item()

            # --- 統計情報の収集 ---
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
