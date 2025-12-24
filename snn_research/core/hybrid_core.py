# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Feedback Alignment実装版)

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
        
        # フィードバック・アライメント用の固定行列 (Feedback Weights)
        # 出力層の誤差(out)を中間層(hidden)へ戻すためのランダムなパス
        # 勾配計算は不要なので requires_grad=False
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
            r = self.deep_process(f) # deep_processはパススルーでも機能する
            out = self.output_gate(r)
            
            reward_scalar = 0.0
            
            if target is not None:
                tgt_idx = target.item()
                out_vec = out.view(-1)
                
                # --- 1. 出力層の誤差計算 (Target - Output) ---
                # 正解は1.0, 不正解は0.0を目指す
                target_vec = torch.zeros_like(out_vec)
                target_vec[tgt_idx] = 1.0
                
                # 単純な差分誤差 (Error Vector)
                # outputが発火していない(0)なら、正解箇所は 1.0 - 0.0 = +1.0 (もっと発火せよ)
                # outputが間違って発火(1)なら、不正解箇所は 0.0 - 1.0 = -1.0 (抑制せよ)
                error_signal = target_vec - out_vec
                
                # 出力層への報酬（指導信号）は、この誤差信号そのものを使うのが最も効率的
                # 前回の「正解なら+1.5」ロジックと同等以上の効果がある
                # さらに学習を加速するためゲインを掛ける
                output_reward = error_signal * 2.0 
                
                # --- 2. フィードバック・アライメント (Hidden層への指導) ---
                # 出力の誤差を、固定されたランダム行列を通して中間層へ投影する
                # これにより、中間層の各ニューロンは「今の誤差を減らすために自分がどうすべきか」を知る
                # (hidden, out) @ (out, 1) -> (hidden, 1)
                hidden_feedback = torch.matmul(self.feedback_matrix, error_signal.unsqueeze(1)).view(-1)
                
                # 正規化: 信号が大きすぎると重みが発散するので、tanhなどで整えるか、スケーリングする
                hidden_reward = torch.tanh(hidden_feedback) * 2.0

                # スカラ評価 (ログ表示用)
                if out_vec[tgt_idx] > 0.5:
                    reward_scalar = 1.0
                else:
                    reward_scalar = -1.0
            else:
                output_reward = 0.0
                hidden_reward = 0.0
                reward_scalar = 0.0

            # 学習の実行
            # Hidden層にも「ベクトル信号」を渡すことで、前回実装した「強制学習モード」を作動させる
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=hidden_reward)
            
            # Output層
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=output_reward)
            
            surprise = float(self.deep_process.last_error.abs().mean().item()) if self.deep_process.last_error is not None else 0.0
            
        return {
            "prediction_error": surprise,
            "reward": float(reward_scalar),
            "output_spike_count": float(out.sum().item()),
            "proficiency": float(self.output_gate.proficiency.item())
        }
