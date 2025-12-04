# ファイルパス: snn_research/cognitive_architecture/planner_snn.py
# (修正: 次元不整合の修正)
# Title: プランナー用SNNモデル
# Description: 
#   エージェントの計画立案やスキル選択を行うためのSNNモデル。
#   Transformerライクな構造を持ちつつ、SNNコアを利用して時系列処理を行う。
#
# 修正:
#   - skill_head の入力次元を d_model ではなく d_state * num_layers に修正。
#     (Predictive Codingモデルの隠れ状態出力仕様に合わせるため)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union, Tuple
from snn_research.core.snn_core import SNNCore

class PlannerSNN(nn.Module):
    """
    プランニングタスクに特化したSNNモデル。
    Predictive Codingアーキテクチャを使用し、内部状態からスキルを選択する。
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        d_state: int, 
        num_layers: int, 
        time_steps: int, 
        n_head: int, 
        num_skills: int, 
        neuron_config: Dict[str, Any]
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.num_skills = num_skills
        self.time_steps = time_steps

        # SNNCore を使用してバックボーンを構築
        # (Predictive Coding型を使用する設定例)
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

        # スキル選択用の出力ヘッド
        # SNNの出力 (Batch, Time, Features) からスキルを選択
        
        # --- ▼ 修正: 入力次元の計算を変更 ▼ ---
        # Predictive Coding (BreakthroughSNN) の output_hidden_states=True 時の出力は
        # 各層の状態(d_state)を結合したものになるため、次元数は d_state * num_layers となる。
        feature_dim = d_state * num_layers
        self.skill_head = nn.Linear(feature_dim, num_skills)
        # --- ▲ 修正 ▲ ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen) の入力ID
        Returns:
            logits: (Batch, NumSkills) のスキル選択確率
        """
        # SNNコアによる処理
        # output_hidden_states=True により (Batch, Seq, feature_dim) が返る
        outputs = self.core(x, output_hidden_states=True)

        # SNNCore (BreakthroughSNNなど) は (logits, spikes, mem) のタプルを返す場合がある
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 時間方向・シーケンス方向のプーリング (平均)
        # outputs: (Batch, SeqLen, feature_dim) または (Time, Batch, SeqLen, feature_dim)
        
        if outputs.dim() == 4: # (Time, Batch, SeqLen, Features)
            pooled = outputs.mean(dim=[0, 2]) # TimeとSeqLenで平均 -> (Batch, Features)
        elif outputs.dim() == 3: # (Batch, SeqLen, Features)
            pooled = outputs.mean(dim=1) # SeqLenで平均 -> (Batch, Features)
        else:
            # 想定外だが、(Batch, Features) の場合はそのまま
            pooled = outputs 

        # pooled: (Batch, feature_dim) -> (Batch, num_skills)
        return self.skill_head(pooled)