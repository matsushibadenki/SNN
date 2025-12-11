# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.5 [Type Fixed]
# Description:
#   LiquidAssociationCortex の初期化時に不足していた num_text_inputs を追加。

from typing import Dict, Any, List, Optional, Union, cast
import time
import logging
import torch
from torchvision import transforms  # type: ignore

# Core Modules
from snn_research.core.snn_core import SNNCore
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

# --- 追加: 統合知覚野 ---
from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex

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
    Artificial Brain Kernel v14.5
    """
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        motivation_system: IntrinsicMotivationSystem,
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        thinking_engine: SNNCore,
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
        logger.info("🚀 Booting Artificial Brain Kernel v14.5 (Neuro-Symbolic OS)...")
        
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        self.thinking_engine = thinking_engine
        
        if astrocyte_network is None:
            self.astrocyte = AstrocyteNetwork()
        else:
            self.astrocyte = astrocyte_network
        
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        
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
        
        # --- 新規追加: Liquid Association Cortex (Unified Perception) ---
        # 修正: num_text_inputs 引数を追加 (256次元と仮定)
        self.association_cortex = LiquidAssociationCortex(
            num_visual_inputs=64,  # VisualCortexの特徴次元
            num_audio_inputs=64,   # 音声特徴次元 (以前のコードのコメントに合わせて調整)
            num_text_inputs=256,   # 言語埋め込み次元
            num_somato_inputs=10,
            reservoir_size=512
        )
        
        self.cycle_count = 0
        self.state = "AWAKE"
        self.energy_level = 100.0
        self.fatigue_level = 0.0
        
        self.current_priming_signal: Optional[torch.Tensor] = None
        
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.astrocyte.request_resource("thinking_engine", 50.0) 
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する。
        """
        if self.state in ["SLEEPING", "DREAMING"]:
            self.astrocyte.request_resource("system_idle", 0.1)
            return {"status": "sleeping", "response": "Zzz..."}

        self.cycle_count += 1
        
        report: Dict[str, Any] = {
            "cycle": self.cycle_count,
            "input": str(raw_input)[:50],
            "executed_modules": [],
            "denied_modules": [],
            "thought_process": None
        }

        if self.current_priming_signal is not None:
            pass

        # --- Step 1: Perception (Unified & Modal) ---
        sensory_info = self.receptor.receive(raw_input)
        
        visual_spikes_for_lac: Optional[torch.Tensor] = None
        audio_spikes_for_lac: Optional[torch.Tensor] = None
        
        if sensory_info['type'] == 'image':
            if self.astrocyte.request_resource("visual_cortex", 15.0):
                img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
                self.visual.perceive_and_upload(img_tensor)
                report["executed_modules"].append("visual_cortex")
                
                visual_spikes_for_lac = torch.rand(1, 64) > 0.9
                
                if self.astrocyte.request_resource("symbol_grounding", 5.0):
                    vis_data = self.workspace.get_information("visual_cortex")
                    if vis_data and "features" in vis_data:
                        concept_id = self.grounding.ground_neural_pattern(vis_data["features"], "visual_input")
                        report["grounded_concept"] = concept_id
                        report["executed_modules"].append("symbol_grounding")
            else:
                report["denied_modules"].append("visual_cortex")
        else:
            if self.astrocyte.request_resource("perception", 2.0):
                spike_pattern = self.encoder.encode(sensory_info, duration=16)
                self.perception.perceive_and_upload(spike_pattern)
                report["executed_modules"].append("perception")
                
                audio_spikes_for_lac = spike_pattern.float().mean(dim=0).unsqueeze(0) > 0.5
                
                if self.astrocyte.request_resource("amygdala", 1.0):
                    content_str = str(sensory_info['content'])
                    self.amygdala.evaluate_and_upload(content_str)
                    report["executed_modules"].append("amygdala")
            else:
                report["denied_modules"].append("perception")

        # ★ Unified Perception: Liquid Association Cortex ★
        lac_vis = visual_spikes_for_lac.float() if visual_spikes_for_lac is not None else None
        lac_aud = audio_spikes_for_lac.float() if audio_spikes_for_lac is not None else None
        
        # text_spikes 等は現状None
        association_activity = self.association_cortex(
            visual_spikes=lac_vis,
            audio_spikes=lac_aud
        )
        
        if self.astrocyte.request_resource("association", 5.0):
            self.workspace.upload_to_workspace(
                source="association_cortex",
                data={"features": association_activity, "type": "integrated_sensation"},
                salience=0.3 
            )
            report["executed_modules"].append("association_cortex")

        # --- Step 2: Consciousness (意識) ---
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: High-Level Cognition ---
        if conscious_content:
            if self.astrocyte.request_resource("thinking_engine", 20.0):
                try:
                    device = next(self.thinking_engine.parameters()).device
                    dummy_ids = torch.randint(0, 1000, (1, 16)).to(device)
                    _ = self.thinking_engine(dummy_ids)
                    thought_output = "[Neural Activity Generated]" 
                    report["thought_process"] = thought_output
                    report["executed_modules"].append("thinking_engine")
                    self.workspace.upload_to_workspace("thinking_engine", thought_output, salience=0.6)
                except Exception as e:
                    logger.warning(f"Thinking engine error: {e}")
            else:
                report["denied_modules"].append("thinking_engine")

            if self.astrocyte.request_resource("prefrontal_cortex", 8.0):
                self.pfc.handle_conscious_broadcast("workspace", conscious_content)
                report["executed_modules"].append("prefrontal_cortex")

            if self.astrocyte.request_resource("causal_inference", 10.0):
                self.causal_engine.handle_conscious_broadcast("workspace", conscious_content)
                report["executed_modules"].append("causal_inference")

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
            
            if self.astrocyte.request_resource("hippocampus", 4.0):
                episode = {
                    "timestamp": time.time(),
                    "input": str(raw_input),
                    "consciousness": conscious_content,
                    "thought": report.get("thought_process"),
                    "action": selected_action if 'selected_action' in locals() else None,
                    "emotion": amygdala_state
                }
                self.hippocampus.store_episode(episode)
                report["executed_modules"].append("hippocampus")

            if isinstance(conscious_content, dict) and "detected_objects" in conscious_content:
                concept = f"neural_concept_{self.cycle_count}"
                priming = self.grounding.recall_pattern(concept)
                if priming is not None:
                    self.current_priming_signal = priming
                    logger.info(f"🔮 Top-Down Priming generated for concept '{concept}'")
            else:
                self.current_priming_signal = None

        # --- Step 4: System Check ---
        self.astrocyte.step() 
        self.energy_level = self.astrocyte.current_energy
        self.fatigue_level = self.astrocyte.fatigue_toxin

        if self.astrocyte.fatigue_toxin > 100.0 or self.astrocyte.current_energy < 50.0:
            logger.info(f"🥱 Brain limit reached. Initiating Sleep...")
            self.sleep_cycle()

        report["energy"] = self.astrocyte.current_energy
        report["fatigue"] = self.astrocyte.fatigue_toxin
        
        return report

    def sleep_cycle(self) -> Dict[str, Any]:
        self.state = "SLEEPING"
        print("\n🌙 --- SLEEP CYCLE INITIATED ---")
        
        phases = []
        syn_change_total = 0.0
        
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        if episodes:
            print(f"   📝 Consolidating {len(episodes)} episodes...")
            for ep in episodes:
                self.cortex.consolidate_memory(ep)
            phases.append(f"Consolidated {len(episodes)} episodes")
        
        if self.sleep_manager:
            self.state = "DREAMING"
            self.astrocyte.request_resource("dreaming", 30.0)
            
            print("   🦄 Dreaming: Generating replay...")
            dream_stats = self.sleep_manager.perform_sleep_cycle()
            
            dreams_count = dream_stats.get('dreams_replayed', 0)
            syn_change = dream_stats.get('synaptic_change', 0.0)
            dream_phases = dream_stats.get('phases', [])
            
            syn_change_total = syn_change
            print(f"   🧠 Replayed {dreams_count} dreams. Phases: {dream_phases}")
            phases.extend(dream_phases)
        
        self.astrocyte.clear_fatigue(100.0) 
        self.astrocyte.replenish_energy(1000.0)
        self.energy_level = self.astrocyte.current_energy
        self.fatigue_level = self.astrocyte.fatigue_toxin
        
        self.state = "AWAKE"
        print(f"🌅 --- WAKE UP (Energy: {self.energy_level:.1f}) ---")
        
        return {"status": "slept", "phases": phases, "synaptic_change": syn_change_total}

    def sleep_and_dream(self):
        self.sleep_cycle()

    def correct_knowledge(self, concept: str, correct_info: str, reason: str = "user_correction"):
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
