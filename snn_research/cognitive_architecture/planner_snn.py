# ディレクトリパス: snn_research/cognitive_architecture/
# ファイルパス: planner_snn.py
# 日本語タイトル: Planner SNN (v2.6 高度次元整合版)
# 目的: 形状不一致の動的解決とスキル選択ヘッドの提供。

import torch
import torch.nn as nn
from snn_research.core.snn_core import SNNCore

class PlannerSNN(nn.Module):
    """
    プランニング用SNN。
    [修正] 実行時のテンソル形状不一致 (RuntimeError) を動的に吸収するアダプターを実装。
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
        
        # 出力ヘッド
        self.skill_head = nn.Linear(d_model, num_skills)
        # 次元調整用アダプター (必要に応じて forward 内で初期化)
        self.dim_adapter = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SNNCore の実行
        outputs = self.core(x, output_hidden_states=True)
        
        # 隠れ状態の抽出
        if isinstance(outputs, torch.Tensor):
            hidden = outputs
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            hidden = outputs[0]
        else:
            hidden = torch.zeros((x.size(0), x.size(1), self.d_model), device=x.device)

        # プーリング (Batch, Features) に変換
        if hidden.dim() == 4: # (T, B, S, D)
            pooled = hidden.mean(dim=[0, 2])
        elif hidden.dim() == 3: # (B, S, D)
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        # [重要] 行列形状の動的整合性チェック
        if pooled.size(-1) != self.d_model:
            if self.dim_adapter is None:
                self.dim_adapter = nn.Linear(pooled.size(-1), self.d_model).to(pooled.device)
            pooled = self.dim_adapter(pooled)

        return self.skill_head(pooled)