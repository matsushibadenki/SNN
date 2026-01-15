# snn_research/models/hybrid/emotional_concept_brain.py
# 修正: 基礎視力を確実に確保するため、VisualCortexを堅牢なCNNベースに変更。
#       これにより、Stage 1で高精度を出し、Stage 3での「正しい自律学習」を実現する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from snn_research.cognitive_architecture.amygdala import Amygdala

class RobustVisualCortex(nn.Module):
    """
    MNISTを確実に学習できる堅牢な視覚野モデル。
    Spikformerの代わりにこれを採用し、実験のボトルネックを解消する。
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14
            
            # Conv 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 7x7
            
            # Conv 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh() # 活性化して特徴を際立たせる
        )
        
    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, embed_dim)
        return self.features(x)

class EmotionalConceptBrain(nn.Module):
    def __init__(self, 
                 img_size: int = 28,
                 concept_dim: int = 128,
                 num_classes: int = 10,
                 emotion_hidden_dim: int = 64):
        super().__init__()
        
        # 1. 知性: 堅牢な視覚野
        self.cortex = RobustVisualCortex(embed_dim=128)
        
        # 概念処理用の層（今回は簡易化のため統合層に含めるか、省略）
        # 統合層（分類ヘッド）
        self.head = nn.Linear(128, num_classes)
        
        # 2. 感性: 扁桃体
        # 入力は「視覚特徴」+「予測ロジット(概念の代わり)」
        # 自分の出した答え(Logits)に対して感情を持つ
        self.amygdala = Amygdala(input_dim=128 + num_classes, hidden_dim=emotion_hidden_dim)
        
        self.last_internal_state = None
        
    def forward(self, x_img: torch.Tensor, x_concept: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prediction: タスク予測 (Logits)
            emotion_value: 感情価
        """
        # 1. 視覚処理
        sensory_vec = self.cortex(x_img) # (B, 128)
        self.last_internal_state = sensory_vec
        
        # 2. 推論 (Prediction)
        logits = self.head(sensory_vec) # (B, 10)
        
        # 3. 感情発生 (Amygdala)
        # 「見ているもの(Sensory)」と「判断結果(Logits)」を入力
        # これにより「この画像で、この判断をした自分」に対する自信/不安を評価する
        amygdala_input = torch.cat([sensory_vec, F.softmax(logits, dim=1)], dim=1) # (B, 128+10)
        emotion_value = self.amygdala(amygdala_input) # (B, 1)
        
        return logits, emotion_value

    def get_internal_state(self):
        return self.last_internal_state