# ファイルパス: snn_research/social/theory_of_mind.py
# 日本語タイトル: Theory of Mind (ToM) Module v1.0
# 目的・内容:
#   ROADMAP Phase 2.4 "Social Intelligence" 対応。
#   他者の行動観測データ（位置、視線、発話など）から、
#   そのエージェントの「隠された意図（Goal/Intent）」を推論するモジュール。
#   高速なBitSpikeMambaを用いて、リアルタイムに相手の心を読み取る。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple

# 高速推論のためMambaを利用
try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None

logger = logging.getLogger(__name__)


class TheoryOfMindEncoder(nn.Module):
    """
    心の理論（ToM）エンコーダ。
    他者の行動シーケンスを入力とし、その意図（Intent Vector）を出力する。
    """

    def __init__(
        self,
        input_dim: int,  # 観測次元 (例: 相手の座標 x,y + 速度 vx,vy = 4)
        hidden_dim: int = 64,
        intent_dim: int = 8,  # 予測する意図のクラス数や座標次元
        model_type: str = "mamba"  # 'mamba' or 'lstm'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_type = model_type

        logger.info(
            f"🧠 Initializing Theory of Mind (ToM) Engine... (Type: {model_type})")

        # 1. Sequence Modeler (Trajectory -> Latent)
        if model_type == "mamba" and BitSpikeMamba is not None:
            # 時系列パターン認識にMambaを使用
            self.core = BitSpikeMamba(
                vocab_size=0,  # Continuous input
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                num_layers=2,
                time_steps=16,  # ヒストリー長
                neuron_config={"type": "lif"}
            )
            self.input_proj = nn.Linear(input_dim, hidden_dim)

        else:
            # Fallback to LSTM/GRU if Mamba not available
            if model_type == "mamba":
                logger.warning("BitSpikeMamba not found. Falling back to GRU.")
            self.core = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True
            )
            self.input_proj = nn.Identity()

        # 2. Intent Decoder (Latent -> Intent/Goal)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intent_dim)
        )

    def forward(self, observation_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation_sequence: [Batch, Time, Input_Dim]
            (e.g., past 10 steps of another agent's position)

        Returns:
            predicted_intent: [Batch, Intent_Dim]
            (e.g., predicted target coordinates)
        """
        B, T, D = observation_sequence.shape

        # Feature Projection
        x = self.input_proj(observation_sequence)  # [B, T, Hidden]

        # Sequence Modeling
        if isinstance(self.core, nn.GRU):
            out, _ = self.core(x)
            final_state = out[:, -1, :]  # Last hidden state
        else:
            # Mamba Forward
            # Mamba returns (logits/features, spikes, mem)
            mamba_out = self.core(x)
            if isinstance(mamba_out, tuple):
                features = mamba_out[0]
            else:
                features = mamba_out

            # Use the feature at the last time step
            if features.dim() == 3:
                final_state = features[:, -1, :]
            else:
                # If T=1 or collapsed
                final_state = features

        # Decode Intent
        intent = self.intent_head(final_state)

        return intent

    def predict_goal(self, trajectory: torch.Tensor) -> torch.Tensor:
        """推論用ラッパー"""
        self.eval()
        with torch.no_grad():
            return self.forward(trajectory)
