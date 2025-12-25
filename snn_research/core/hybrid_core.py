# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: マージン学習ロジック)

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

class TopKActivation(nn.Module):
    def __init__(self, sparsity: float = 0.25, gain: float = 2.0) -> None:
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
        self.activation = TopKActivation(sparsity=0.25, gain=2.0)
        
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

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r) # Logits (Membrane Potential)
            
            # SNNの出力は通常スパイク(0/1)だが、学習のために膜電位(Current)を参照する
            # output_gate.forward内で membrane_potential に平均電位が格納されているが、
            # ここでは学習用に `out` (スパイク) とは別に、内部状態(current)を使いたい。
            # しかし LogicGatedSNN の forward は spikes を返す。
            # 学習には spikes (out) と Error (target) を使う。
            
            loss_val = 0.0
            acc = 0.0
            
            if target is not None:
                # --- マージン最大化誤差計算 ---
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                # 通常の誤差: (正解 - 出力)
                # マージン強化: 正解ラベルに対しては、出力が弱ければ強いプラスの誤差を与える
                # 不正解ラベルに対しては、出力が強ければ強いマイナスの誤差を与える
                
                # 出力がスパイク(0,1)なので、離散的。
                # 正解なのにスパイクしてない(0) -> Error +1 (Update positive)
                # 不正解なのにスパイクしてる(1) -> Error -1 (Update negative)
                
                # ここに「確信度」を加えるため、膜電位(membrane_potential)を参照したいが、
                # バッチ処理の簡略化のためスパイクベースで行く。
                # 代わりに「過剰学習」を許容する（Errorをブーストする）。
                
                error = (target_onehot - out) * 1.5 # 誤差信号を増幅
                
                # さらに、正解ラベルの膜電位が低い場合、スパイクしていても強化する
                # （ここがマージン学習の肝だが、内部変数へのアクセスが複雑になるため、
                #  今回はシンプルにError増幅で対応）

                self.output_gate.update_plasticity(r, out, reward=error)

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
