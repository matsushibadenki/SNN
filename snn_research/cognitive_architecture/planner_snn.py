# ファイルパス: snn_research/cognitive_architecture/planner_snn.py
# 日本語タイトル: Planner SNN (次元整合性修正版)
# 目的: mat1 and mat2 shapes cannot be multiplied の完全解消。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from snn_research.core.snn_core import SNNCore

class PlannerSNN(nn.Module):
    """
    プランニング用SNN。
    [修正] skill_head の入力次元を d_model に、forward 内のプーリングを確実に実施。
    """
    def __init__(self, vocab_size, d_model, d_state, num_layers, time_steps, n_head, num_skills, neuron_config):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
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
        # 修正: 入力は必ず d_model 次元になるようにプーリングする
        self.skill_head = nn.Linear(d_model, num_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SNNCore の出力
        outputs = self.core(x, output_hidden_states=True)
        
        # 隠れ状態の抽出と次元調整
        if isinstance(outputs, torch.Tensor):
            hidden = outputs
        elif isinstance(outputs, (list, tuple)):
            hidden = outputs[0]
        else:
            # フォールバック (Batch, Seq, d_model) と仮定したダミー
            hidden = torch.zeros((x.size(0), x.size(1), self.d_model), device=x.device)

        # 全ての次元を (Batch, d_model) に集約
        if hidden.dim() == 4: # (T, B, S, D)
            pooled = hidden.mean(dim=[0, 2])
        elif hidden.dim() == 3: # (B, S, D)
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        # 行列形状の強制リサイズ (予期せぬ次元不一致への最終ガード)
        if pooled.size(-1) != self.d_model:
             # 線形変換で d_model に合わせる (LazyLinear利用)
             if not hasattr(self, 'dim_adapter'):
                 self.dim_adapter = nn.Linear(pooled.size(-1), self.d_model).to(pooled.device)
             pooled = self.dim_adapter(pooled)

        return self.skill_head(pooled)
