# snn_research/models/hybrid/emotional_concept_brain.py
# 目的: 知性(Concept)と感性(Emotion)を統合した脳モデル。
#       自分の推論結果に対して感情(Value)を抱き、それを元に学習する。

import torch
import torch.nn as nn
from typing import Optional, Tuple

from snn_research.models.hybrid.concept_spikformer import ConceptSpikformer
from snn_research.cognitive_architecture.amygdala import Amygdala

class EmotionalConceptBrain(nn.Module):
    def __init__(self, 
                 img_size: int = 28,
                 concept_dim: int = 128,
                 num_classes: int = 10,
                 emotion_hidden_dim: int = 64):
        super().__init__()
        
        # 1. 知性: 概念理解を行う脳 (ConceptSpikformer)
        self.cortex = ConceptSpikformer(
            img_size=img_size,
            embed_dim=128,
            concept_dim=concept_dim,
            num_classes=num_classes,
            projection_dim=64 # Contrastive用
        )
        
        # 2. 感性: 状態の価値判断を行う扁桃体 (Amygdala)
        # 入力は「視覚特徴」と「概念特徴」の結合
        self.amygdala = Amygdala(input_dim=128 + 128, hidden_dim=emotion_hidden_dim)
        
    def forward(self, x_img: torch.Tensor, x_concept: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prediction: タスク予測結果
            emotion_value: その状態に対する感情価 (-1.0 ~ 1.0)
        """
        # 1. 視覚処理 (Bottom-up)
        sensory_rep = self.cortex.forward_sensory(x_img) # (B, N, 128)
        sensory_vec = sensory_rep.mean(dim=1) # (B, 128)
        
        # 2. 概念処理 (Top-down)
        if x_concept is not None:
            # 外部から概念が与えられた場合（教師あり学習時など）
            concept_rep = self.cortex.forward_conceptual(x_concept) # (B, 128)
        else:
            # 自律動作時: 自分の予測を概念として扱う（思考ループ）
            # ここでは簡易的にゼロベクトルまたは直前の思考を使う
            concept_rep = torch.zeros_like(sensory_vec) 

        # 3. 統合と推論
        prediction = self.cortex.integrate(sensory_rep, x_concept) # (B, 128) -> (B, Classes)
        
        # 4. 感情発生 (Amygdala)
        # 「見ているもの(Sensory)」と「考えていること(Concept)」を入力し、
        # その整合性や過去の情動記憶に基づいて「価値」を判断する
        amygdala_input = torch.cat([sensory_vec, concept_rep], dim=1) # (B, 256)
        emotion_value = self.amygdala(amygdala_input) # (B, 1)
        
        return prediction, emotion_value

    def get_internal_state(self):
        return self.cortex.get_internal_state()