# ファイルパス: snn_research/systems/autonomous_learning_loop.py
# 日本語タイトル: Autonomous Learning Loop v2.1 (Fixes for Sleep API)
# 目的・内容:
#   SleepConsolidator v2.2 (Hippocampusベース) に対応。
#   Hippocampusを初期化し、エピソード記憶を経由して睡眠学習を行うように修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional
import logging

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

logger = logging.getLogger(__name__)


class AutonomousLearningLoop:
    """
    自律学習ライフサイクル管理システム (v2.1)
    """

    def __init__(
        self,
        agent: EmbodiedVLMAgent,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        energy_capacity: float = 1000.0,
        fatigue_threshold: float = 800.0
    ):
        self.device = device
        self.agent = agent.to(device)
        self.optimizer = optimizer

        logger.info(f"⚙️ Initializing AutonomousLearningLoop on {device}...")

        # Phase 2 Components
        self.motivator = IntrinsicMotivationSystem().to(device)

        # 記憶システムの初期化 (v2.2 Architecture)
        self.cortex = Cortex()
        self.hippocampus = Hippocampus(rag_system=self.cortex.rag_system)

        # Sleep Consolidator (Hippocampus -> Brain/Cortex)
        # Note: optimizer is re-initialized inside SleepConsolidator for specific params if needed,
        # but here we might pass None or handle it inside.
        # The new SleepConsolidator signature: (hippocampus, cortex, target_brain_model, ...)
        self.sleep_system = SleepConsolidator(
            hippocampus=self.hippocampus,
            cortex=self.cortex,
            target_brain_model=self.agent,
            device=device
        )

        # World Predictor
        if hasattr(agent, "motor_decoder"):
            fusion_dim = agent.motor_decoder.input_dim
        else:
            fusion_dim = getattr(agent, "fusion_dim", 512)
        action_dim = getattr(agent, "action_dim", 64)

        self.world_predictor = nn.Sequential(
            nn.Linear(fusion_dim + action_dim, 512),
            nn.GELU(),
            nn.Linear(512, fusion_dim)
        ).to(device)

        self.predictor_optimizer = optim.AdamW(
            self.world_predictor.parameters(), lr=1e-3)

        # Homeostasis
        self.energy = energy_capacity
        self.max_energy = energy_capacity
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold

    def step(self,
             current_image: torch.Tensor,
             current_text: torch.Tensor,
             next_image: Optional[torch.Tensor] = None
             ) -> Dict[str, Any]:

        # 1. Sleep Check
        if self._should_sleep():
            return self._perform_sleep_cycle()

        self.agent.train()
        self.world_predictor.train()

        # 2. Perception & Action
        agent_out = self.agent(current_image, current_text)
        z_t = agent_out.get("fused_context")
        action = agent_out.get("action_pred")

        # 3. Prediction
        if z_t is not None and action is not None:
            if z_t.dim() > 2:
                z_t = z_t.mean(dim=1)
            pred_input = torch.cat([z_t, action], dim=-1)
            z_next_pred = self.world_predictor(pred_input)
        else:
            z_next_pred = torch.zeros(1, 512).to(self.device)

        # 4. Surprise Calculation
        surprise = 0.0
        prediction_loss = torch.tensor(0.0).to(self.device)

        if next_image is not None:
            with torch.no_grad():
                next_out = self.agent.vlm(next_image, current_text)
                z_next_actual = next_out.get("fused_representation")
                if z_next_actual is not None:
                    if z_next_actual.dim() > 2:
                        z_next_actual = z_next_actual.mean(dim=1)
                    prediction_loss = F.mse_loss(z_next_pred, z_next_actual)
                    surprise = torch.clamp(prediction_loss, 0.0, 1.0).item()

        # 5. Motivation
        self.motivator.process(input_payload=z_t, prediction_error=surprise)
        intrinsic_reward = self.motivator.calculate_intrinsic_reward(
            surprise=surprise)

        # 6. Memory Storage (Hippocampus)
        # 辞書形式のエピソードとして保存
        episode = {
            "input": current_image.detach().cpu(),  # CPUに退避してメモリ節約
            "text": current_text.detach().cpu(),
            "reward": intrinsic_reward,
            "surprise": surprise
        }
        self.hippocampus.process(episode)

        # 7. Learning
        total_loss = prediction_loss + agent_out.get("alignment_loss", 0) * 0.1
        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.predictor_optimizer.step()

        # 8. Homeostasis
        self.energy -= 1.0
        self.fatigue += (0.5 + surprise * 2.0)

        drives = self.motivator.update_drives(
            surprise=surprise,
            energy_level=self.energy,
            fatigue_level=self.fatigue,
            task_success=True
        )

        return {
            "mode": "wake",
            "loss": total_loss.item(),
            "surprise": surprise,
            "energy": self.energy,
            "drives": drives
        }

    def _should_sleep(self) -> bool:
        if self.fatigue >= self.fatigue_threshold or self.energy <= 0:
            return True
        return False

    def _perform_sleep_cycle(self) -> Dict[str, Any]:
        # SleepConsolidator v2.2 API
        report = self.sleep_system.perform_sleep_cycle(duration_cycles=5)

        self.fatigue = 0.0
        self.energy = self.max_energy * 0.9

        return {
            "mode": "sleep",
            "report": report,
            "energy": self.energy
        }
