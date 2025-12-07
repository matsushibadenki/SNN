# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.3 (Plasticity Integrated)
# Description:
#   ロードマップ Phase 5 & 7 に基づく、人工脳のオペレーティングシステム実装。
#   修正: 睡眠サイクルの実行と、CorticalColumnの可塑性を統合。

from typing import Dict, Any, List, Optional, Union, cast
import time
import logging
import torch
from torchvision import transforms # type: ignore

# Core Modules
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

# Cognitive Modules
from .visual_perception import VisualCortex
from .hybrid_perception_cortex import HybridPerceptionCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus
from .cortex import Cortex
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex
from .causal_inference_engine import CausalInferenceEngine
from .intrinsic_motivation import IntrinsicMotivationSystem
from .symbol_grounding import SymbolGrounding
from .sleep_consolidation import SleepConsolidator

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    Artificial Brain Kernel v14.3
    自律的にエネルギーを管理し、学習（覚醒）と整理（睡眠）を繰り返すニューロシンボリックOS。
    """
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        motivation_system: IntrinsicMotivationSystem,
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        perception_cortex: HybridPerceptionCortex,
        visual_cortex: VisualCortex,
        prefrontal_cortex: PrefrontalCortex,
        hippocampus: Hippocampus,
        cortex: Cortex,
        amygdala: Amygdala,
        basal_ganglia: BasalGanglia,
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex,
        causal_inference_engine: CausalInferenceEngine,
        symbol_grounding: SymbolGrounding,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None
    ):
        logger.info("🚀 Booting Artificial Brain Kernel v14.3 (OS Mode)...")
        
        # --- Kernel Core ---
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        
        if astrocyte_network is None:
            self.astrocyte = AstrocyteNetwork()
        else:
            self.astrocyte = astrocyte_network
        
        # --- IO Interfaces ---
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        
        # --- Cognitive Modules ---
        self.perception = perception_cortex
        self.visual = visual_cortex
        self.pfc = prefrontal_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        self.causal_engine = causal_inference_engine
        self.grounding = symbol_grounding
        
        # --- System State ---
        self.cycle_count = 0
        self.state = "AWAKE"
        
        # 状態パラメータ（可視化用）
        self.energy_level = 100.0 # Legacy property for dashboard
        self.fatigue_level = 0.0  # Legacy property for dashboard
        
        # Utilities
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する。
        """
        # 1. OS状態チェック
        if self.state in ["SLEEPING", "DREAMING"]:
            self.astrocyte.request_resource("system_idle", 0.1)
            return {"status": "sleeping", "response": "Zzz..."}

        self.cycle_count += 1
        
        # レポート用コンテナ
        report: Dict[str, Any] = {
            "cycle": self.cycle_count,
            "input": str(raw_input)[:50],
            "executed_modules": [],
            "denied_modules": []
        }

        # --- Step 1: Perception (入力処理) ---
        sensory_info = self.receptor.receive(raw_input)
        
        # 視覚処理 (高コスト)
        if sensory_info['type'] == 'image':
            if self.astrocyte.request_resource("visual_cortex", 15.0):
                img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
                self.visual.perceive_and_upload(img_tensor)
                report["executed_modules"].append("visual_cortex")
                
                # Grounding
                if self.astrocyte.request_resource("symbol_grounding", 5.0):
                    vis_data = self.workspace.get_information("visual_cortex")
                    if vis_data and "features" in vis_data:
                        concept_id = self.grounding.ground_neural_pattern(vis_data["features"], "visual_input")
                        report["grounded_concept"] = concept_id
                        report["executed_modules"].append("symbol_grounding")
            else:
                report["denied_modules"].append("visual_cortex")
        
        # テキスト/一般処理 (低コスト)
        else:
            if self.astrocyte.request_resource("perception", 2.0):
                # HybridPerceptionCortexの処理
                spike_pattern = self.encoder.encode(sensory_info, duration=16)
                self.perception.perceive_and_upload(spike_pattern)
                report["executed_modules"].append("perception")
                
                # Amygdala (重要・高優先度)
                if self.astrocyte.request_resource("amygdala", 1.0):
                    content_str = str(sensory_info['content'])
                    self.amygdala.evaluate_and_upload(content_str)
                    report["executed_modules"].append("amygdala")
            else:
                report["denied_modules"].append("perception")

        # --- Step 2: Consciousness (意識) ---
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: High-Level Cognition (高次認知) ---
        if conscious_content:
            # PFC
            if self.astrocyte.request_resource("prefrontal_cortex", 8.0):
                self.pfc.handle_conscious_broadcast("workspace", conscious_content)
                report["executed_modules"].append("prefrontal_cortex")

            # Causal Inference
            if self.astrocyte.request_resource("causal_inference", 10.0):
                self.causal_engine.handle_conscious_broadcast("workspace", conscious_content)
                report["executed_modules"].append("causal_inference")

            # Basal Ganglia & Motor
            amygdala_state = self.workspace.get_information("amygdala")
            if self.astrocyte.request_resource("basal_ganglia", 3.0):
                selected_action = self.basal_ganglia.select_action(
                    action_candidates=[
                        {'action': 'reply_text', 'value': 0.8}, 
                        {'action': 'store_memory', 'value': 0.6},
                        {'action': 'ignore', 'value': 0.1}
                    ],
                    emotion_context=amygdala_state
                )
                report["executed_modules"].append("basal_ganglia")
                
                if selected_action:
                    action_name = selected_action.get('action')
                    report["action"] = action_name
                    
                    if self.astrocyte.request_resource("motor_cortex", 10.0):
                        motor_commands = self.cerebellum.refine_action_plan(selected_action)
                        execution_log = self.motor.execute_commands(motor_commands)
                        self.actuator.run_command_sequence(execution_log)
                        report["executed_modules"].append("motor_cortex")
            
            # Hippocampus (エピソード記憶)
            if self.astrocyte.request_resource("hippocampus", 4.0):
                episode = {
                    "timestamp": time.time(),
                    "input": str(raw_input),
                    "consciousness": conscious_content,
                    "action": selected_action if 'selected_action' in locals() else None,
                    "emotion": amygdala_state
                }
                self.hippocampus.store_episode(episode)
                report["executed_modules"].append("hippocampus")

        # --- Step 4: System Check ---
        self.astrocyte.step() # 恒常性維持
        
        # Dashboard用プロパティの更新
        self.energy_level = self.astrocyte.current_energy
        self.fatigue_level = self.astrocyte.fatigue_toxin

        # 睡眠導入判定
        if self.astrocyte.fatigue_toxin > 100.0 or self.astrocyte.current_energy < 50.0:
            print(f"🥱 Brain limit reached. Initiating Sleep...")
            self.sleep_cycle()

        report["energy"] = self.astrocyte.current_energy
        report["fatigue"] = self.astrocyte.fatigue_toxin
        
        return report

    def sleep_cycle(self) -> Dict[str, Any]:
        """
        睡眠サイクル。疲労回復、エネルギー補充、記憶固定化を行う。
        """
        self.state = "SLEEPING"
        print("\n🌙 --- SLEEP CYCLE INITIATED ---")
        
        phases = []
        
        # Phase 1: Explicit Consolidation (海馬 -> 皮質GraphRAG)
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        if episodes:
            print(f"   📝 Consolidating {len(episodes)} episodes to Cortex (GraphRAG)...")
            for ep in episodes:
                self.cortex.consolidate_memory(ep)
            phases.append(f"Consolidated {len(episodes)} episodes")
        
        # Phase 2: Generative Replay (Dreaming -> SNN Plasticity)
        if self.sleep_manager:
            self.state = "DREAMING"
            self.astrocyte.request_resource("dreaming", 20.0)
            
            print("   🦄 Dreaming: Generating replay from knowledge graph...")
            dream_stats = self.sleep_manager.perform_sleep_cycle()
            
            dreams_count = dream_stats.get('dreams_replayed', 0)
            syn_change = dream_stats.get('synaptic_change', 0.0)
            print(f"   🧠 Neural Plasticity: Replayed {dreams_count} concepts, Synaptic Delta: {syn_change:.4f}")
            phases.append(f"Dreamed {dreams_count} concepts")
        else:
            logger.warning("   ⚠️ SleepManager not attached. Skipping generative replay.")
        
        # Phase 3: Restoration
        self.astrocyte.clear_fatigue(80.0)
        self.astrocyte.replenish_energy(500.0)
        self.energy_level = self.astrocyte.current_energy
        self.fatigue_level = self.astrocyte.fatigue_toxin
        
        self.state = "AWAKE"
        print(f"🌅 --- WAKE UP (Energy: {self.energy_level:.1f}, Fatigue: {self.fatigue_level:.1f}) ---")
        
        return {"status": "slept", "phases": phases, "synaptic_change": locals().get('syn_change', 0.0)}

    def sleep_and_dream(self):
        """外部からの強制睡眠"""
        self.sleep_cycle()

    def correct_knowledge(self, concept: str, correct_info: str, reason: str = "user_correction"):
        """知識の訂正（Phase 5 Real-time Knowledge Editing）"""
        if self.astrocyte.request_resource("cortex", 5.0):
            logger.info(f"🛠️ Knowledge Correction: '{concept}' -> '{correct_info}'")
            if self.cortex.rag_system:
                self.cortex.rag_system.update_knowledge(
                    subj=concept, 
                    pred="is_corrected_to", 
                    new_obj=correct_info, 
                    reason=reason
                )
        else:
            logger.warning("⚠️ Cannot correct knowledge now: Brain too tired.")
