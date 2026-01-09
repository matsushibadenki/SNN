# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain v2.2 (Dependency Injection Support)
# 目的: Containers.pyからの依存注入に対応し、引数エラーを解消する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

# Cognitive Modules (Imports for type hinting/default instantiation)
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.visual_perception import VisualPerception
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

logger = logging.getLogger(__name__)


class ArtificialBrain(nn.Module):
    """
    SNNプロジェクトの中核となる統合脳モデル (Brain v2.2)。
    DIコンテナからの注入に対応し、柔軟な構成を可能にする。
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        # DI Components (Injected from BrainContainer)
        global_workspace: Optional[GlobalWorkspace] = None,
        motivation_system: Optional[IntrinsicMotivationSystem] = None,
        sensory_receptor: Optional[SensoryReceptor] = None,
        spike_encoder: Optional[SpikeEncoder] = None,
        actuator: Optional[Actuator] = None,
        thinking_engine: Optional[nn.Module] = None,  # Core SNN/Transformer
        perception_cortex: Optional[Any] = None,  # HybridPerceptionCortex
        visual_cortex: Optional[VisualPerception] = None,
        prefrontal_cortex: Optional[PrefrontalCortex] = None,
        hippocampus: Optional[Hippocampus] = None,
        cortex: Optional[Cortex] = None,
        amygdala: Optional[Amygdala] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        cerebellum: Optional[Cerebellum] = None,
        motor_cortex: Optional[MotorCortex] = None,
        causal_inference_engine: Optional[CausalInferenceEngine] = None,
        symbol_grounding: Optional[SymbolGrounding] = None,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.config = config or {}

        logger.info("🧠 Initializing ArtificialBrain v2.2 (DI Supported)...")

        # --- 1. Core Systems & Memory ---
        self.cortex = cortex or Cortex()
        self.hippocampus = hippocampus or Hippocampus(
            rag_system=self.cortex.rag_system,
            short_term_capacity=self.config.get("stm_capacity", 50)
        )
        self.motivation = motivation_system or IntrinsicMotivationSystem(
            curiosity_weight=self.config.get("curiosity_weight", 1.0)
        )

        # --- 2. Cognitive Control ---
        self.global_workspace = global_workspace or GlobalWorkspace()
        self.pfc = prefrontal_cortex or PrefrontalCortex(
            workspace=self.global_workspace,
            motivation_system=self.motivation
        )

        # --- 3. Perception & Action ---
        self.visual_cortex = visual_cortex or VisualPerception(
            num_neurons=self.config.get("input_neurons", 784),
            feature_dim=self.config.get("feature_dim", 256),
            workspace=self.global_workspace
        )
        self.motor_cortex = motor_cortex or MotorCortex()
        self.sensory_receptor = sensory_receptor
        self.spike_encoder = spike_encoder
        self.actuator = actuator

        # --- 4. Advanced Cognition ---
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.causal_engine = causal_inference_engine
        self.symbol_grounding = symbol_grounding
        self.astrocyte_network = astrocyte_network
        self.perception_cortex = perception_cortex

        # --- 5. Core Engine & Maintenance ---
        self.core_model = thinking_engine

        # Sleep Consolidator
        self.sleep_consolidator = sleep_consolidator or SleepConsolidator(
            memory_system=None,
            hippocampus=self.hippocampus,
            cortex=self.cortex,
            target_brain_model=self.core_model,
            dream_rate=self.config.get("dream_rate", 0.1)
        )
        # Ensure model link if injected separately
        if self.core_model and self.sleep_consolidator.brain_model is None:
            self.sleep_consolidator.brain_model = self.core_model

        # State Variables
        self.is_sleeping = False
        self.energy_level = 100.0
        self.step_count = 0

    def set_core_model(self, model: nn.Module):
        """学習対象のコアモデル（SNNなど）をセットし、SleepConsolidatorにも紐付ける"""
        self.core_model = model
        self.sleep_consolidator.brain_model = model
        logger.info(f"🧠 Core brain model set: {type(model).__name__}")

    def process_step(self, sensory_input: Any, reward: float = 0.0) -> Dict[str, Any]:
        """
        1タイムステップの脳活動サイクルを実行。
        知覚 -> 動機評価 -> 意思決定 -> 行動 -> 記憶
        """
        self.step_count += 1

        # 0. 睡眠判定
        internal_state = self.motivation.get_internal_state()
        if self.should_sleep(internal_state):
            return self.perform_sleep_cycle()

        # 1. 知覚 (Perception)
        visual_features = None
        # 入力がTensorまたはRaw Dataの場合の処理
        if self.spike_encoder and not isinstance(sensory_input, torch.Tensor):
            # Encode if needed (Mock logic)
            pass

        if isinstance(sensory_input, torch.Tensor):
            perception_output = self.visual_cortex.perceive(sensory_input)
            visual_features = perception_output.get("features")

        # 2. 動機付け更新 (Motivation)
        surprise = 0.1  # Dummy value for now
        motivation_status = self.motivation.process(
            sensory_input, prediction_error=surprise)
        intrinsic_reward = self.motivation.calculate_intrinsic_reward(
            surprise, external_reward=reward)

        # 3. 記憶 (Memory Encoding)
        episode = {
            "step": self.step_count,
            "input": str(sensory_input)[:50],
            "reward": reward,
            "surprise": surprise,
            "internal_state": motivation_status
        }
        self.hippocampus.process(episode)

        # 4. 意識のブロードキャスト (GWT)
        conscious_content = self.global_workspace.broadcast(
            inputs=[visual_features, episode],
            context=self.pfc.current_goal
        )

        # 5. 行動選択 (Action)
        # PFC -> Motorの流れ (簡易版)
        action_plan = self.pfc.plan(conscious_content)
        action_cmd = self.motor_cortex.generate_command(
            action_plan or conscious_content)

        if self.actuator:
            self.actuator.execute(action_cmd)

        # エネルギー消費
        self.energy_level -= 0.1

        return {
            "action": action_cmd,
            "motivation": motivation_status,
            "intrinsic_reward": intrinsic_reward,
            "conscious_content": "active",
            "is_sleeping": False
        }

    def should_sleep(self, internal_state: Dict[str, float]) -> bool:
        """睡眠に入るべきか判断する"""
        if self.energy_level < 20.0:
            return True
        if internal_state.get("boredom", 0.0) > 0.9:
            return True
        return False

    def perform_sleep_cycle(self, cycles: int = 5) -> Dict[str, Any]:
        """睡眠サイクルを実行し、記憶を整理・定着させる"""
        if self.is_sleeping:
            return {"status": "already_sleeping"}

        self.is_sleeping = True
        logger.info("💤 Entering sleep mode...")

        sleep_report = self.sleep_consolidator.perform_sleep_cycle(
            duration_cycles=cycles)

        self.energy_level = 100.0
        self.is_sleeping = False
        logger.info("🌅 Waking up. Energy restored.")

        return {
            "action": "sleep",
            "sleep_report": sleep_report,
            "is_sleeping": True
        }

    def forward(self, x):
        """PyTorchのforward互換用 (主に学習時)"""
        return self.process_step(x)
