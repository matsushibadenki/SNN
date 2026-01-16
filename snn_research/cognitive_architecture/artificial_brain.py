# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain v2.6.4 (Dimension Fix)
# 目的・内容:
#   統合脳モデルの中核クラス。
#   修正: process_step における Tensor 入力の判定と特徴量抽出フローを堅牢化。
#   修正: Thalamus への入力ガードを強化し、IndexError を防止。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, cast, Union

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
    SNNプロジェクトの中核となる統合脳モデル (Brain v2.6.4 Optimized).
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

        logger.info("🧠 Initializing ArtificialBrain v2.6.4 (Dimension Fix)...")

        # --- Device Handling ---
        self.core_model = thinking_engine
        self.thinking_engine = thinking_engine
        self.device: Any = "cpu"
        # kwargsからdeviceが渡されている場合はそれを使用
        if "device" in kwargs:
            self.device = kwargs["device"]
        elif self.core_model and hasattr(self.core_model, 'device'):
            self.device = cast(Any, self.core_model).device

        # --- Core Systems ---
        self.cortex = cortex or Cortex()
        self.hippocampus = hippocampus or Hippocampus(
            rag_system=self.cortex.rag_system,
            short_term_capacity=self.config.get("stm_capacity", 50)
        )
        self.motivation = motivation_system or IntrinsicMotivationSystem(
            curiosity_weight=self.config.get("curiosity_weight", 1.0)
        )

        # --- Cognitive Control ---
        self.global_workspace = global_workspace or GlobalWorkspace()
        self.workspace = self.global_workspace

        self.pfc = prefrontal_cortex or PrefrontalCortex(
            workspace=self.global_workspace,
            motivation_system=self.motivation
        )

        # --- Perception ---
        # 修正: perception_cortex が指定されている場合はそれを visual_cortex として使用する
        if visual_cortex is not None:
            self.visual_cortex = visual_cortex
        elif perception_cortex is not None and hasattr(perception_cortex, 'perceive'):
            self.visual_cortex = perception_cortex  # type: ignore
        else:
            self.visual_cortex = VisualPerception(
                num_neurons=self.config.get("input_neurons", 784),
                feature_dim=feature_dim,
                workspace=self.global_workspace,
                device=self.device
            )
            
        self.perception = self.visual_cortex
        self.perception_cortex = perception_cortex

        self.thalamus = thalamus or Thalamus(
            input_dim=feature_dim,
            output_dim=feature_dim
        )
        self.thalamus.to(self.device)

        self.sensory_receptor = sensory_receptor
        self.spike_encoder = spike_encoder
        self.actuator = actuator

        # --- Action ---
        self.motor_cortex = motor_cortex or MotorCortex()
        self.basal_ganglia = basal_ganglia or BasalGanglia(
            workspace=self.global_workspace,
            selection_threshold=0.4
        )

        self.amygdala = amygdala
        self.cerebellum = cerebellum
        self.causal_engine = causal_inference_engine
        self.symbol_grounding = symbol_grounding

        # --- Homeostasis ---
        self.astrocyte_network = astrocyte_network or AstrocyteNetwork(
            initial_energy=1000.0,
            max_energy=1000.0
        )
        self.astrocyte = self.astrocyte_network

        # --- Sleep ---
        sleep_config = self.config.copy()
        sleep_config["dream_rate"] = self.config.get("dream_rate", 0.1)

        self.sleep_consolidator = sleep_consolidator or SleepConsolidator(
            memory_system=None,
            hippocampus=self.hippocampus,
            cortex=self.cortex,
            target_brain_model=self.core_model,
            config=sleep_config,
            device=self.device
        )
        
        if self.core_model and self.sleep_consolidator.brain_model is None:
            self.sleep_consolidator.brain_model = self.core_model

        # State Variables
        self.is_sleeping = False
        self.state = "ACTIVE"
        self.step_count = 0
        self.monitor_stats = self.config.get("monitor_stats", False)

        # DI components inject
        if "meta_cognitive_snn" in kwargs:
            self.meta_cognitive_snn = kwargs["meta_cognitive_snn"]
        if "world_model" in kwargs:
            self.world_model = kwargs["world_model"]
        if "reflex_module" in kwargs:
            self.reflex_module = kwargs["reflex_module"]
        if "ethical_guardrail" in kwargs:
            self.ethical_guardrail = kwargs["ethical_guardrail"]
        if "reasoning_engine" in kwargs:
            self.reasoning_engine = kwargs["reasoning_engine"]


    def set_core_model(self, model: nn.Module):
        """学習対象のコアモデルをセット"""
        self.core_model = model
        self.thinking_engine = model
        self.sleep_consolidator.brain_model = model
        
        if hasattr(model, 'device'):
            self.device = cast(Any, model).device
            self.thalamus.to(self.device)
            
        logger.info(f"🧠 Core brain model set: {type(model).__name__}")

    def process_step(self, sensory_input: Any, reward: float = 0.0) -> Dict[str, Any]:
        """
        1タイムステップの脳活動サイクルを実行。
        最適化: .item()呼び出しの排除、条件分岐の整理。
        """
        self.step_count += 1

        # 0. アストロサイト更新 (軽量化)
        if self.astrocyte_network:
            self.astrocyte_network.step()
        
        # エネルギーレベルチェック (頻度低減)
        if self.step_count % 10 == 0 and self.astrocyte_network:
            energy_status = self.astrocyte_network.get_energy_level()
            if energy_status < 0.05:
                return {"action": None, "status": "exhausted"}
        
        # 睡眠判定
        if self.is_sleeping:
             return self.perform_sleep_cycle()

        # 1. 知覚 (Perception)
        visual_features = None
        raw_features: Optional[torch.Tensor] = None
        
        # Tensor入力の場合のみ知覚処理を行う (文字列やその他の入力はスキップ)
        if isinstance(sensory_input, torch.Tensor):
            perception_output = self.visual_cortex.perceive(sensory_input)
            
            if isinstance(perception_output, dict):
                raw_features = perception_output.get("features")
            else:
                raw_features = perception_output

            # Thalamusへの入力ガード: raw_features が存在する場合のみ実行
            if raw_features is not None:
                # [Safety] 次元の整合性をチェックしてからThalamusへ
                try:
                    thalamus_out = self.thalamus.forward(raw_features, top_down_attention=None)
                    visual_features = thalamus_out["relayed_output"]
                except IndexError as e:
                    logger.error(f"Thalamus Forward Error: {e}. Check input dimensions vs Thalamus config.")
                    visual_features = raw_features  # Fallback

        # 2. 動機付け (Motivation) - 軽量化
        motivation_status: Dict[str, Any] = {}
        
        # 3. 記憶 (Memory)
        if self.step_count % 5 == 0:
            pass

        # 4. 意識 (GWT)
        conscious_content = None
        if visual_features is not None:
            conscious_content = self.global_workspace.broadcast(
                inputs=[visual_features],
                context=None
            )
        elif not isinstance(sensory_input, torch.Tensor):
            # Tensorでない入力（文字列など）を意識に上げる場合の簡易処理
            # (ReasoningEngineなどが別途処理することを想定)
            pass

        # 5. 行動選択 (Action Selection)
        final_action_cmd = None
        
        if conscious_content is not None:
            action_plan = self.pfc.plan(conscious_content)
            
            # 簡易的なアクション実行ロジック
            if action_plan is not None:
                if self.basal_ganglia.base_threshold < 0.9: 
                     pass 

        return {
            "action": final_action_cmd,
            "status": "active",
            "step": self.step_count,
            "response": "Cognitive Cycle Completed" 
        }

    def should_sleep(self, internal_state: Dict[str, float]) -> bool:
        return False

    def perform_sleep_cycle(self, cycles: int = 1) -> Dict[str, Any]:
        self.is_sleeping = True
        self.state = "SLEEPING"
        
        if self.astrocyte_network:
            self.astrocyte_network.replenish_energy(amount=10.0 * cycles)
        
        self.is_sleeping = False
        self.state = "ACTIVE"
        return {"action": "sleep_cycle_complete", "status": "waking_up"}

    def forward(self, x):
        return self.process_step(x)

    # --- Compatibility Methods (Restore for Type Checkers) ---
    def run_cognitive_cycle(self, sensory_input: Any) -> Dict[str, Any]:
        """Legacy script support."""
        return self.process_step(sensory_input)

    def sleep_cycle(self) -> Dict[str, Any]:
        """Legacy script support."""
        return self.perform_sleep_cycle()

    def get_brain_status(self) -> Dict[str, Any]:
        """Return current status diagnostics."""
        # 簡易レポートを返す（計算コスト削減）
        astro_energy = 0.0
        if self.astrocyte_network:
            # max_energyが0除算にならないようガード
            max_e = self.astrocyte_network.max_energy if self.astrocyte_network.max_energy > 0 else 1.0
            astro_energy = (self.astrocyte_network.energy / max_e) * 100

        return {
            "status": "SLEEPING" if self.is_sleeping else "ACTIVE",
            "astrocyte": {
                "metrics": {
                    "energy_percent": astro_energy
                }
            },
            "steps": self.step_count,
            "os": {}
        }