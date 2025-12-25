# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: Soft k-WTA & Signal Gain)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class TopKActivation(nn.Module):
    """
    k-Winner-Take-All Activation with Gain:
    上位k%のみを通し、さらに信号を増幅してSNNの発火を促進する。
    """
    def __init__(self, sparsity: float = 0.25, gain: float = 2.0) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # k個のニューロンを選定
        k = int(x.shape[1] * self.sparsity)
        if k < 1: k = 1
        
        # 上位k個の値とインデックスを取得
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        # マスク作成
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_indices, 1.0)
        
        # 信号を通過させ、ゲイン倍して次層のSNN閾値(1.0)を超えやすくする
        return x * mask * self.gain

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # 【修正】sparsity 0.1 -> 0.25: より多くの情報を流す
        # 【追加】gain=2.0: 信号を増幅
        self.activation = TopKActivation(sparsity=0.25, gain=2.0)
        
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
            r = self.deep_process(f) 
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
