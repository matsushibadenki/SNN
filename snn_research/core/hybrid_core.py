# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: Reward-Modulated Hebbian版)

import torch
import torch.nn as nn
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

# 簡易的なパススルー層
class ActivePredictiveLayer(nn.Module):
    def __init__(self, features): super().__init__()
    def forward(self, x): return x

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        # フィードバック行列
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
            # Forward
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar_val = 0.0
            
            if target is not None:
                # 誤差信号 (Target - Output)
                # 正解なら正の報酬、不正解なら負の報酬
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                # 単純な誤差信号
                error = (target_onehot - out)
                
                # 出力層の更新: 教師あり学習 (Delta Rule)
                # エラーが大きい方向に重みを動かす
                self.output_gate.update_plasticity(r, out, reward=error)
                
                # 隠れ層の更新: Feedback Alignment
                # 出力の誤差を隠れ層へ逆投影し、それを「報酬」として学習する
                # これにより、出力の誤差を減らすような特徴抽出を隠れ層が行うようになる
                hidden_reward = torch.matmul(error, self.feedback_matrix)
                
                self.fast_process.update_plasticity(x_input, f, reward=hidden_reward)

                # ログ
                pred = out.argmax(dim=1)
                reward_scalar_val = (pred == target).float().mean().item()

        return {
            "prediction_error": 0.0,
            "reward": reward_scalar_val,
            "output_spike_count": float(out.sum().item() / x_input.size(0)),
            "proficiency": float(self.output_gate.proficiency.item())
        }
