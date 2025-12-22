# ファイルパス: snn_research/cognitive_architecture/planner_snn.py
# 日本語タイトル: Planner SNN (次元不一致修正版)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from snn_research.core.snn_core import SNNCore

class PlannerSNN(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, num_layers, time_steps, n_head, num_skills, neuron_config):
        super().__init__()
        self.d_model = d_model
        # バックボーン
        self.core = SNNCore(
            config={
                'architecture_type': 'predictive_coding', 
                'd_model': d_model,
                'num_layers': num_layers,
                'time_steps': time_steps,
                'neuron': neuron_config,
                'd_state': d_state, 
                'n_head': n_head    
            },
            vocab_size=vocab_size
        )
        # [修正] d_model を入力次元として固定
        self.skill_head = nn.Linear(d_model, num_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output_hidden_states=True を利用して隠れ状態を取得
        hidden_state = self.core(x, output_hidden_states=True)
        
        # 形状を (Batch, d_model) にプーリング
        if isinstance(hidden_state, torch.Tensor):
            if hidden_state.dim() == 4: # (T, B, S, D)
                pooled = hidden_state.mean(dim=[0, 2])
            elif hidden_state.dim() == 3: # (B, S, D)
                pooled = hidden_state.mean(dim=1)
            else:
                pooled = hidden_state
        else:
            # フォールバック
            pooled = torch.zeros((x.size(0), self.d_model), device=x.device)

        return self.skill_head(pooled)
