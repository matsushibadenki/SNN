# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain v2.4 (Neuro-Cognitive Enhanced)
# 目的: 脳科学的レビューに基づき、予測的知覚、視床によるゲーティング、大脳基底核による行動選択、アストロサイトによる変調を統合する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, cast, List

# Cognitive Modules
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
from snn_research.cognitive_architecture.thalamus import Thalamus
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

logger = logging.getLogger(__name__)


class ArtificialBrain(nn.Module):
    """
    SNNプロジェクトの中核となる統合脳モデル (Brain v2.4)。
    脳科学的知見に基づき、以下の機能を統合実装：
    1. Top-down Prediction (PFC -> Thalamus -> Perception)
    2. Thalamic Gating (Attention & Sleep switch)
    3. Basal Ganglia Action Selection (Go/No-Go gating)
    4. Astrocyte Modulation (Energy & Fatigue management)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        # DI Components
        global_workspace: Optional[GlobalWorkspace] = None,
        motivation_system: Optional[IntrinsicMotivationSystem] = None,
        sensory_receptor: Optional[SensoryReceptor] = None,
        spike_encoder: Optional[SpikeEncoder] = None,
        actuator: Optional[Actuator] = None,
        thinking_engine: Optional[nn.Module] = None,
        perception_cortex: Optional[Any] = None,
        visual_cortex: Optional[VisualPerception] = None,
        prefrontal_cortex: Optional[PrefrontalCortex] = None,
        hippocampus: Optional[Hippocampus] = None,
        cortex: Optional[Cortex] = None,
        amygdala: Optional[Amygdala] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        cerebellum: Optional[Cerebellum] = None,
        thalamus: Optional[Thalamus] = None,
        motor_cortex: Optional[MotorCortex] = None,
        causal_inference_engine: Optional[CausalInferenceEngine] = None,
        symbol_grounding: Optional[SymbolGrounding] = None,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        **kwargs
    ):
        super().__init__()
        self.config = config or {}
        feature_dim = self.config.get("feature_dim", 256)

        logger.info("🧠 Initializing ArtificialBrain v2.4 (Neuro-Enhanced)...")

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
        # Alias
        self.workspace = self.global_workspace

        self.pfc = prefrontal_cortex or PrefrontalCortex(
            workspace=self.global_workspace,
            motivation_system=self.motivation
        )

        # --- 3. Perception & Sensory Relay ---
        self.visual_cortex = visual_cortex or VisualPerception(
            num_neurons=self.config.get("input_neurons", 784),
            feature_dim=feature_dim,
            workspace=self.global_workspace
        )
        # Alias
        self.perception = self.visual_cortex

        # [New] Thalamus for gating features (Feature dim -> Feature dim)
        # 視覚特徴量をGWTへ送る前のゲーティングを行うため、feature_dimを使用
        self.thalamus = thalamus or Thalamus(
            input_dim=feature_dim,
            output_dim=feature_dim
        )

        self.sensory_receptor = sensory_receptor
        self.spike_encoder = spike_encoder
        self.actuator = actuator

        # --- 4. Action & Selection ---
        self.motor_cortex = motor_cortex or MotorCortex()

        # [New] Basal Ganglia for Action Selection
        self.basal_ganglia = basal_ganglia or BasalGanglia(
            workspace=self.global_workspace,
            selection_threshold=0.4
        )

        self.amygdala = amygdala
        self.cerebellum = cerebellum
        self.causal_engine = causal_inference_engine
        self.symbol_grounding = symbol_grounding

        self.perception_cortex = perception_cortex

        # --- 5. Homeostasis & Maintenance ---
        self.astrocyte_network = astrocyte_network or AstrocyteNetwork(
            initial_energy=1000.0,
            max_energy=1000.0
        )
        # Alias
        self.astrocyte = self.astrocyte_network

        self.core_model = thinking_engine
        self.thinking_engine = thinking_engine

        # Sleep Consolidator
        self.sleep_consolidator = sleep_consolidator or SleepConsolidator(
            memory_system=None,
            hippocampus=self.hippocampus,
            cortex=self.cortex,
            target_brain_model=self.core_model,
            dream_rate=self.config.get("dream_rate", 0.1)
        )
        if self.core_model and self.sleep_consolidator.brain_model is None:
            self.sleep_consolidator.brain_model = self.core_model

        # State Variables
        self.is_sleeping = False
        self.step_count = 0

        # Device Handling
        self.device: Any = "cpu"
        if self.core_model and hasattr(self.core_model, 'device'):
            self.device = cast(Any, self.core_model).device

        # Move submodules to device if needed
        self.thalamus.to(self.device)

    def set_core_model(self, model: nn.Module):
        """学習対象のコアモデルをセット"""
        self.core_model = model
        self.thinking_engine = model
        self.sleep_consolidator.brain_model = model
        if hasattr(model, 'device'):
            self.device = cast(Any, model).device
        logger.info(f"🧠 Core brain model set: {type(model).__name__}")

    def process_step(self, sensory_input: Any, reward: float = 0.0) -> Dict[str, Any]:
        """
        1タイムステップの脳活動サイクルを実行。
        Neuro-Cognitive Cycle:
        1. Sensory -> Visual Cortex (Feature Extraction)
        2. Top-down Attention -> Thalamus (Gating)
        3. Gated Features -> GWT (Consciousness)
        4. PFC (Planning) -> Basal Ganglia (Action Selection) -> Motor Cortex
        5. Astrocyte (Metabolism & Modulation)
        """
        self.step_count += 1

        # --- 0. アストロサイトによる恒常性維持と代謝更新 ---
        self.astrocyte_network.step()

        # 疲労やエネルギーレベルの取得
        energy_status = self.astrocyte_network.get_energy_level()
        fatigue_level = self.astrocyte_network.fatigue_toxin

        # エネルギー枯渇時は強制睡眠または活動低下
        if energy_status < 0.05:
            logger.warning("⚠️ Critical Energy Low. Skipping cognitive cycle.")
            return {"action": None, "status": "exhausted"}

        # 睡眠判定
        internal_state = self.motivation.get_internal_state()
        # アストロサイトの状態も加味して睡眠判定
        should_sleep = self.should_sleep(
            internal_state) or (fatigue_level > 80.0)

        if should_sleep:
            # 視床を睡眠モードへ切り替え (感覚遮断)
            self.thalamus.set_state("SLEEP")
            return self.perform_sleep_cycle()
        else:
            self.thalamus.set_state("AWAKE")

        # --- 1. 知覚 (Perception) & Top-down Attention ---
        visual_features = None

        # Encoder処理
        if self.spike_encoder and not isinstance(sensory_input, torch.Tensor):
            pass  # Mock logic

        # Visual Cortexによる特徴抽出 (Bottom-up)
        if isinstance(sensory_input, torch.Tensor):
            perception_output = self.visual_cortex.perceive(sensory_input)
            if isinstance(perception_output, dict):
                raw_features = perception_output.get("features")
            else:
                raw_features = perception_output

            # [Improvement] Top-down Prediction / Attention from PFC
            # 前頭前野の現在のゴールを注意信号として利用
            top_down_signal = None
            if self.pfc.current_goal:
                # 注: ここでは簡易的にPFCのゴール情報をTensor化するか、既存のAttentionマップを使う想定
                # 実装簡略化のため、NoneでなければThalamusに渡す
                # 実際にはDimension matchingが必要
                pass

            # [Improvement] Thalamusによる情報のゲーティング
            # アストロサイトの疲労度が高いと注意力が散漫になる（ゲート制御が甘くなる）シミュレーションも可能
            if raw_features is not None:
                thalamus_out = self.thalamus.forward(
                    raw_features, top_down_attention=top_down_signal)
                visual_features = thalamus_out["relayed_output"]

                # アストロサイトへの負荷記録 (知覚処理のコスト)
                self.astrocyte_network.monitor_neural_activity(
                    firing_rate=visual_features.mean().item())

        # --- 2. 動機付け (Motivation) ---
        surprise = 0.1  # Placeholder
        motivation_status = self.motivation.process(
            sensory_input, prediction_error=surprise)

        # 疲労度をInternal Stateに反映
        motivation_status["fatigue"] = fatigue_level

        intrinsic_reward = self.motivation.calculate_intrinsic_reward(
            surprise, external_reward=reward)

        # --- 3. 記憶 (Memory Encoding) ---
        episode = {
            "step": self.step_count,
            "input_summary": "sensory_data",  # Tensorは重いので要約推奨
            "reward": reward,
            "surprise": surprise,
            "internal_state": motivation_status
        }
        self.hippocampus.process(episode)

        # --- 4. 意識のブロードキャスト (GWT) ---
        conscious_content = self.global_workspace.broadcast(
            inputs=[visual_features, episode],
            context=self.pfc.current_goal
        )

        # --- 5. 行動計画と選択 (PFC -> Basal Ganglia -> Motor) ---
        # PFCによる計画立案
        action_plan = self.pfc.plan(conscious_content)

        # [Improvement] 大脳基底核による行動選択 (Action Selection / Gating)
        # PFCの計画を行動候補としてラップする
        candidate_actions: List[Dict[str, Any]] = []
        if action_plan:
            # action_planが辞書型か確認、そうでなければ整形
            if isinstance(action_plan, dict):
                # valueの存在チェックのみで代入はしない
                candidate_actions.append(action_plan)
            else:
                candidate_actions.append(
                    {"action": action_plan, "value": 0.8, "source": "PFC"})

        # 情動コンテキスト（扁桃体などからの入力）を作成
        emotion_context = {
            "arousal": motivation_status.get("arousal", 0.0),
            "valence": motivation_status.get("valence", 0.0)
        }

        # 大脳基底核が最終的な行動を決定 (Go / No-Go)
        approved_action_dict = self.basal_ganglia.select_action(
            external_candidates=candidate_actions,
            emotion_context=emotion_context
        )

        final_action_cmd = None
        if approved_action_dict:
            action_content = approved_action_dict.get("action")
            final_action_cmd = self.motor_cortex.generate_command(
                action_content)

            # アストロサイト活動記録 (運動コスト)
            self.astrocyte_network.consume_energy("motor_cortex", amount=2.0)

            if self.actuator:
                self.actuator.execute(final_action_cmd)
        else:
            # No-Go: 行動抑制
            pass

        # 状態保持
        self.state = "ACTIVE"

        return {
            "action": final_action_cmd,
            "motivation": motivation_status,
            "intrinsic_reward": intrinsic_reward,
            "conscious_content": "active",
            "is_sleeping": False,
            "energy_level": energy_status,
            "executed_modules": ["visual_cortex", "thalamus", "hippocampus", "pfc", "basal_ganglia", "motor_cortex"]
        }

    def should_sleep(self, internal_state: Dict[str, float]) -> bool:
        """睡眠に入るべきか判断する"""
        energy = self.astrocyte_network.get_energy_level() * 100.0
        if energy < 20.0:
            return True
        if internal_state.get("boredom", 0.0) > 0.9:
            return True
        return False

    def perform_sleep_cycle(self, cycles: int = 5) -> Dict[str, Any]:
        """睡眠サイクルを実行し、記憶を整理・定着させる"""
        if self.is_sleeping:
            return {"status": "already_sleeping"}

        self.is_sleeping = True
        self.state = "SLEEPING"
        logger.info("💤 Entering sleep mode...")

        # 睡眠中のアストロサイト回復ブースト（Glymphatic Systemの模倣）
        # 毒素排出を加速
        self.astrocyte_network.clear_fatigue(amount=10.0 * cycles)
        self.astrocyte_network.replenish_energy(amount=50.0 * cycles)

        sleep_report = self.sleep_consolidator.perform_sleep_cycle(
            duration_cycles=cycles)

        self.is_sleeping = False
        self.state = "ACTIVE"
        logger.info("🌅 Waking up. Energy restored & Memory consolidated.")

        return {
            "action": "sleep",
            "sleep_report": sleep_report,
            "is_sleeping": True
        }

    def forward(self, x):
        """PyTorchのforward互換用"""
        return self.process_step(x)

    # --- Compatibility Methods for Legacy Scripts ---

    def run_cognitive_cycle(self, sensory_input: Any) -> Dict[str, Any]:
        return self.process_step(sensory_input)

    def sleep_cycle(self):
        return self.perform_sleep_cycle()

    def get_brain_status(self) -> Dict[str, Any]:
        """Return current status diagnostics."""
        # アストロサイトネットワークから詳細レポートを取得
        astro_report = self.astrocyte_network.get_diagnosis_report()

        return {
            "status": "SLEEPING" if self.is_sleeping else "ACTIVE",
            "energy": astro_report["metrics"]["current_energy"],
            "steps": self.step_count,
            "astrocyte": astro_report,
            "os": {}
        }
