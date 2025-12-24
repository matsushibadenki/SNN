# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: 完全バッチ対応版)

import torch
import torch.nn as nn
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        self.register_buffer(
            'feedback_matrix', 
            torch.randn(hidden_features, out_features)
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar_val = 0.0
            
            if target is not None:
                # バッチ処理 (target: [Batch])
                batch_size = target.size(0)
                
                # One-hot target matrix: (Batch, Out)
                target_matrix = torch.zeros_like(out)
                target_matrix.scatter_(1, target.unsqueeze(1), 1.0)
                
                # 誤差信号: (Batch, Out)
                # 正解は1.0, 不正解は0.0
                error_signal = target_matrix - out
                
                # 出力層報酬: (Batch, Out)
                output_reward = error_signal * 2.0 
                
                # フィードバック・アライメント (Hidden層報酬)
                # (Batch, Out) @ (Out, Hidden) -> (Batch, Hidden)
                # error_signal: (B, O), FB: (H, O) -> FB.T: (O, H)
                # matmul(error, FB.T)
                hidden_feedback = torch.matmul(error_signal, self.feedback_matrix.t())
                hidden_reward = torch.tanh(hidden_feedback) * 2.0

                # ログ用スカラ報酬 (平均)
                # 正解クラスの出力が0.5を超えている率
                correct_probs = out.gather(1, target.unsqueeze(1))
                reward_scalar_val = float((correct_probs > 0.5).float().mean().item())
                
            else:
                output_reward = 0.0
                hidden_reward = 0.0
                reward_scalar_val = 0.0

            # 学習の実行 (バッチデータ全体を渡す)
            self.fast_process.update_plasticity(x_input, f, reward=hidden_reward)
            self.output_gate.update_plasticity(r, out, reward=output_reward)
            
            surprise = float(self.deep_process.last_error.abs().mean().item()) if self.deep_process.last_error is not None else 0.0
            
        return {
            "prediction_error": surprise,
            "reward": reward_scalar_val,
            "output_spike_count": float(out.sum().item() / x_input.size(0)), # per sample avg
            "proficiency": float(self.output_gate.proficiency.item())
        }
