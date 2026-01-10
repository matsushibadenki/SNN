# ファイルパス: snn_research/models/motor/spiking_motor_decoder.py
# 日本語タイトル: Spiking Motor Decoder (Neural Action Generator)
# 目的・内容:
#   ROADMAP Phase 2 "Multimodal Integration" 対応。
#   統合された潜在表現（Fused Latents）から、物理的なアクション信号を生成する。
#   LIFニューロンを用いて、スパイク頻度を連続値（モーター出力）や離散値（コマンド）に変換する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

from snn_research.core.factories import NeuronFactory

logger = logging.getLogger(__name__)


class SpikingMotorDecoder(nn.Module):
    """
    高次の概念（スパイク列）を運動指令にデコードするモジュール。
    Continuous Control (ロボットアーム等) と Discrete Control (移動コマンド等) の両方に対応。
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_type: str = "continuous",  # 'continuous' or 'discrete'
        hidden_dim: int = 128,
        time_steps: int = 16,
        neuron_config: Dict[str, Any] = {"type": "lif"}
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.time_steps = time_steps

        # 1. Motor Planning Layer (Hidden State)
        self.plan_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            NeuronFactory.create(features=hidden_dim, config=neuron_config)
        )

        # 2. Motor Execution Layer (Output)
        self.exec_layer = nn.Linear(hidden_dim, action_dim)

        # 出力層のニューロン (連続値制御の場合は積分してアナログ値にするため、LIFを通した後に平均化などが一般的)
        # ここでは最終層はLinear出力とし、損失関数側で制御する構成をとる（Direct Code Prediction）
        # または、Last Layer Spiking -> Low Pass Filter もありだが、学習安定性のためLinear Readoutを採用。

        logger.info(
            f"🦾 SpikingMotorDecoder initialized. Type: {action_type}, Out: {action_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Latents [Batch, Time, Input_Dim] (from VLM Fused Representation)

        Returns:
            action_output: [Batch, Action_Dim] (Continuous values or Logits)
        """
        B, T, D = x.shape

        # 時間方向の情報を集約しつつ、運動計画を生成
        # SNN的には各ステップで処理し、最後にReadoutする

        # Reset neurons if stateful (assuming NeuronFactory handles state internally or functional calls)
        # For simplicity in this module, we assume functional or auto-reset in forward loop if implemented
        # Here using simple feedforward for demonstration of structure

        # Input x is already a sequence of features (spikes or embeddings)
        # We process it to extract the "Action Intent"

        # Flatten time or Average pooling based on strategy
        # Strategy: Temporal Average of Plan Layer -> Execution

        # Apply Plan Layer per step (if applicable) or on aggregated features
        # ここでは「文脈全体から行動を決定する」ため、時間平均をとってからDecodeする
        # （リアルタイム制御の場合はステップごとの出力が必要だが、VLM連携では「判断」が主）

        x_mean = x.mean(dim=1)  # [B, Input_Dim]

        plan = self.plan_layer(x_mean)  # [B, Hidden_Dim] (Spikes/Activation)

        if isinstance(plan, tuple):  # Some neuron models return (spikes, mem)
            plan = plan[0]

        action_out = self.exec_layer(plan)  # [B, Action_Dim]

        if self.action_type == "continuous":
            # Tanh for normalized motor control (-1 to 1)
            action_out = torch.tanh(action_out)

        return action_out
