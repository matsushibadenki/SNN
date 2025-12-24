# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: バッチ対応強化版)

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        # x_input: (batch, in_features)
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar = 0.0
            
            if target is not None:
                # 学習時はバッチサイズ1を想定
                tgt_idx = target.item()
                out_vec = out.view(-1)
                
                # ベースライン報酬
                reward_vector = torch.full_like(out_vec, -0.2)
                
                # 正解ニューロン強化
                reward_vector[tgt_idx] = 1.5
                
                # スカラ報酬の計算
                if out_vec[tgt_idx] > 0.5:
                    wrong_fires = out_vec.sum() - out_vec[tgt_idx]
                    reward_scalar = float(1.0 - (wrong_fires * 0.5))
                else:
                    reward_scalar = -1.0
            else:
                reward_vector = 0.0
                reward_scalar = 0.0

            # 学習の実行
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=reward_scalar)
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=reward_vector)
            
            surprise = float(self.deep_process.last_error.abs().mean().item()) if self.deep_process.last_error is not None else 0.0
            
        return {
            "prediction_error": surprise,
            "reward": float(reward_scalar),
            "output_spike_count": float(out.sum().item()),
            "proficiency": float(self.output_gate.proficiency.item())
        }
