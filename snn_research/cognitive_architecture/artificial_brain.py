# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.0 (Debug Enabled)
# Description:
#   ロードマップ Phase 5 & 7 に基づく、人工脳のオペレーティングシステム実装。
#   修正: run_cognitive_cycle 内にデバッグ用の print(flush=True) を追加し、
#   処理の進行状況を確実にログに出力させる。

from typing import Dict, Any, List, Optional, Union, cast
import time
import logging
import torch
import sys # 追加
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
    Artificial Brain Kernel v14.0
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
        logger.info("🚀 Booting Artificial Brain Kernel v14.0...")
        
        # --- Kernel Core ---
        self.workspace = global_workspace
        self.astrocyte = astrocyte_network
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        
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
        
        # --- System State (Homeostasis) ---
        self.cycle_count = 0
        self.state = "AWAKE"
        self.energy_level = 100.0
        self.fatigue_level = 0.0
        
        # Utilities
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する（覚醒時）。
        感覚 -> 知覚 -> 意識 -> 決定 -> 行動 -> 記憶 -> 恒常性調整
        """
        print("[Brain] Cycle Started", flush=True) # Debug

        # 1. OS状態チェック (睡眠・エネルギー)
        if self.state in ["SLEEPING", "DREAMING"]:
            logger.info("💤 Brain is sleeping... (Input buffered or ignored)")
            return {"status": "sleeping", "response": "Zzz..."}
            
        if self.energy_level <= 5.0 or self.fatigue_level >= 90.0:
            logger.warning("⚠️ Critical Fatigue/Energy levels. Initiating forced sleep.")
            return self.sleep_cycle()

        self.cycle_count += 1
        self._consume_energy(0.5)
        self.fatigue_level += 0.5
        
        report: Dict[str, Any] = {"cycle": self.cycle_count, "input": str(raw_input)[:50]}

        # --- Step 1: Perception & Grounding (入力処理) ---
        print("[Brain] Step 1: Perception", flush=True) # Debug
        sensory_info = self.receptor.receive(raw_input)
        
        if sensory_info['type'] == 'image':
            img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
            self.visual.perceive_and_upload(img_tensor)
            vis_data = self.workspace.get_information("visual_cortex")
            if vis_data and "features" in vis_data:
                concept_id = self.grounding.ground_neural_pattern(vis_data["features"], "visual_input")
                report["grounded_concept"] = concept_id
        else:
            content_str = str(sensory_info['content'])
            spike_pattern = self.encoder.encode(sensory_info, duration=16)
            self.perception.perceive_and_upload(spike_pattern)
            self.amygdala.evaluate_and_upload(content_str)
            self.grounding.process_observation({"text": content_str}, "text_input")

        # --- Step 2: Consciousness (意識の競合とブロードキャスト) ---
        print("[Brain] Step 2: Consciousness", flush=True) # Debug
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: Decision Making & Action (意思決定) ---
        print("[Brain] Step 3: Decision Making", flush=True) # Debug
        if conscious_content:
            self.pfc.handle_conscious_broadcast("workspace", conscious_content)
            
            amygdala_state = self.workspace.get_information("amygdala")
            selected_action = self.basal_ganglia.select_action(
                action_candidates=[
                    {'action': 'reply_text', 'value': 0.8}, 
                    {'action': 'store_memory', 'value': 0.6},
                    {'action': 'ignore', 'value': 0.1}
                ],
                emotion_context=amygdala_state
            )
            
            if selected_action:
                action_name = selected_action.get('action')
                report["action"] = action_name
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                execution_log = self.motor.execute_commands(motor_commands)
                self.actuator.run_command_sequence(execution_log)
                self._consume_energy(2.0)
            
            # --- Step 4: Memory Formation (エピソード記憶) ---
            print("[Brain] Step 4: Memory", flush=True) # Debug
            episode = {
                "timestamp": time.time(),
                "input": str(raw_input),
                "consciousness": conscious_content,
                "action": selected_action,
                "emotion": amygdala_state
            }
            self.hippocampus.store_episode(episode)

        # --- Step 5: Homeostasis (恒常性維持) ---
        print("[Brain] Step 5: Homeostasis", flush=True) # Debug
        if self.astrocyte:
            self.astrocyte.step()

        self._check_fatigue()
        
        report["energy"] = self.energy_level
        report["fatigue"] = self.fatigue_level
        
        print("[Brain] Cycle Finished", flush=True) # Debug
        return report

    def sleep_cycle(self) -> Dict[str, Any]:
        """睡眠サイクルを実行する。"""
        self.state = "SLEEPING"
        logger.info("\n🌙 --- SLEEP CYCLE INITIATED ---")
        logger.info(f"   Initial Fatigue: {self.fatigue_level:.1f}, Energy: {self.energy_level:.1f}")

        phases: List[str] = []
        
        # Phase 1: Explicit Consolidation
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        if episodes:
            logger.info(f"   📝 Consolidating {len(episodes)} episodes to Cortex (GraphRAG)...")
            for ep in episodes:
                self.cortex.consolidate_memory(ep)
            phases.append(f"Consolidated {len(episodes)} episodes")
        
        # Phase 2: Implicit Consolidation
        if self.sleep_manager:
            self.state = "DREAMING"
            dream_stats = self.sleep_manager.perform_sleep_cycle()
            synaptic_change = dream_stats.get('synaptic_change', 0.0)
            dreams_count = dream_stats.get('dreams_replayed', 0)
            logger.info(f"   🦄 Dream Cycle: Replayed {dreams_count} concepts.")
            phases.append(f"Dreamed {dreams_count} concepts")
        else:
            logger.warning("   ⚠️ SleepManager not attached.")
        
        # Phase 3: Restoration
        self.fatigue_level = max(0.0, self.fatigue_level - 80.0)
        self.energy_level = 100.0
        self.state = "AWAKE"
        
        logger.info("🌅 --- WAKE UP ---")
        return {"status": "slept", "phases": phases}

    def _consume_energy(self, amount: float):
        self.energy_level = max(0.0, self.energy_level - amount)
        if self.energy_level < 20.0:
            self.fatigue_level += amount * 2.0

    def _check_fatigue(self):
        memory_load = len(self.hippocampus.working_memory) / self.hippocampus.capacity
        if self.fatigue_level > 80.0 or memory_load >= 1.0:
            logger.info("🥱 Drowsiness detected. Initiating sleep cycle...")
            self.sleep_cycle()
    
    def sleep_and_dream(self):
        self.sleep_cycle()

    def correct_knowledge(self, concept: str, correct_info: str, reason: str = "user_correction"):
        logger.info(f"🛠️ Knowledge Correction Request: '{concept}' -> '{correct_info}'")
        if self.cortex.rag_system:
            self.cortex.rag_system.update_knowledge(
                subj=concept, 
                pred="is_corrected_to", 
                new_obj=correct_info, 
                reason=reason
            )
        correction_episode = {
            "type": "knowledge_correction",
            "concept": concept,
            "content": correct_info,
            "reason": reason,
            "timestamp": time.time()
        }
        self.hippocampus.store_episode(correction_episode)
