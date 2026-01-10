# ファイルパス: snn_research/cognitive_architecture/qualia_synthesizer.py
# 日本語タイトル: Qualia Synthesizer (Subjective Experience Generator) v1.0
# 目的・内容:
#   ROADMAP Phase 4.2 "Synthetic Phenomenology" 対応。
#   客観的な「感覚データ(Sensory Data)」に、主観的な「情動(Affect)」と「身体感覚(Interoception)」を
#   非線形に融合させ、エージェント固有の「クオリア（質感）」ベクトルを生成する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class QualiaSynthesizer(nn.Module):
    """
    感覚と感情を融合し、主観的体験（Qualia）を生成するモジュール。
    """

    def __init__(self, sensory_dim: int = 64, emotion_dim: int = 8, qualia_dim: int = 64):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.emotion_dim = emotion_dim
        self.qualia_dim = qualia_dim

        # 1. Emotional Coloring Layer (感情による味付け)
        # 感覚入力に対して、感情状態が変調（Modulation）をかける
        self.modulation = nn.Sequential(
            nn.Linear(emotion_dim, sensory_dim),
            nn.Sigmoid()  # 0.0 ~ 1.0 のゲート係数
        )

        # 2. Phenomenological Integration (現象学的統合)
        # 変調された感覚と、生の感情を混ぜ合わせて高次元空間へ射影
        self.integration = nn.Sequential(
            nn.Linear(sensory_dim + emotion_dim, qualia_dim * 2),
            nn.GELU(),
            nn.Linear(qualia_dim * 2, qualia_dim),
            nn.LayerNorm(qualia_dim)  # 空間上の位置を安定させる
        )

        logger.info(
            "🌈 Qualia Synthesizer initialized. (Sensory + Affect -> Experience)")

    def forward(self, sensory_input: torch.Tensor, emotion_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sensory_input: 客観的な観測データ [Batch, Dim]
            emotion_state: 内部の情動状態 [Batch, E_Dim] (例: [Fear, Joy, Anger...])

        Returns:
            qualia: 主観的体験ベクトル
        """
        # Batch次元の調整
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)
        if emotion_state.dim() == 1:
            emotion_state = emotion_state.unsqueeze(0)

        # 1. Affective Modulation (感情による着色)
        # 悲しい時は世界が暗く見える、楽しい時は明るく見える等のフィルタリング
        mod_factors = self.modulation(emotion_state)
        colored_sensation = sensory_input * mod_factors

        # 2. Integration
        # 「着色された感覚」と「感情そのもの」を結合
        combined = torch.cat([colored_sensation, emotion_state], dim=-1)

        # クオリア生成
        qualia_vector = self.integration(combined)

        return {
            "qualia": qualia_vector,
            "modulation": mod_factors
        }

    def compute_subjective_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """
        2つの体験の「主観的な違い」を計算する。
        客観的に同じ入力でも、クオリア空間での距離が遠ければ「全く違う体験」として扱われる。
        """
        return (1.0 - F.cosine_similarity(q1, q2)).item()
