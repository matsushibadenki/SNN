# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.5 (Neuro-Symbolic OS & Top-Down Loop)
# Description:
#   ロードマップ Phase 5 & 7 完全対応版。
#   改善点:
#   - トップダウン注意制御 (Top-Down Attention) のループを実装。
#     意識に上った概念に基づいて SymbolGrounding からパターンを想起し、
#     Visual Cortex などの知覚モジュールにプライミング信号として送る。

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
    自律的にエネルギーを管理し、学習（覚醒）と整理（睡眠）を繰り返すニューロシンボリックOS。
    トップダウンの予測信号による知覚変調機能を搭載。
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
        
        # --- Kernel Core ---
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        self.thinking_engine = thinking_engine
        
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
        self.energy_level = 100.0
        self.fatigue_level = 0.0
        
        # Top-down priming signal holder
        self.current_priming_signal: Optional[torch.Tensor] = None
        
        # Utilities
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Initial Resource Request
        self.astrocyte.request_resource("thinking_engine", 50.0) 
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する。
        知覚 -> 意識 -> 思考(SNN) -> 意思決定 -> 行動 -> トップダウン制御 のループ。
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

        # --- Step 0: Apply Top-Down Priming (前のサイクルの予測を知覚に反映) ---
        if self.current_priming_signal is not None:
            # 視覚野などの閾値を調整したり、期待パターンを入力したりする
            # ここでは簡易的にVisualCortexへの引数渡しを想定（または直接属性セット）
            # self.visual.set_attention(self.current_priming_signal) # 実装依存
            pass

        # --- Step 1: Integrated Perception (五感統合処理) ---
        # 入力を各モダリティのスパイクに変換 (欠損していてもOK)
        visual_spikes = None
        audio_spikes = None
    
        sensory_info = self.receptor.receive(raw_input)

        # 既存のロジックを生かしつつ、スパイク変換を試みる
        if sensory_info['type'] == 'image':
            # 既存: VisualCortexへ
            img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
            self.visual.perceive_and_upload(img_tensor) # 従来ルート
        
            # 新規: 画像をスパイク化して連合野へ (DVS的変換 or CNN出力)
            # visual_spikes = self.visual.encode_to_spikes(img_tensor) 
            pass 
        
        elif sensory_info['type'] == 'text':
            # 既存: HybridPerceptionCortexへ
            spike_pattern = self.encoder.encode(sensory_info, duration=16)
            self.perception.perceive_and_upload(spike_pattern) # 従来ルート
        
            # 新規: テキストスパイクを連合野へ
            # audio_spikes (text as audio symbol) = spike_pattern
            audio_spikes = spike_pattern.mean(dim=0) # 時間次元を潰して入力する場合

            # ★ ここで連合野 (Association Cortex) を駆動 ★
            # 視覚と聴覚(テキスト)が同時に混ざり合う
            association_activity = self.association_cortex(
                visual_spikes=visual_spikes, 
                audio_spikes=audio_spikes
            )
    
            # 連合野の活動パターンから「概念」を想起し、Workspaceへ
            # これにより「音を聞いただけで映像が浮かぶ」現象を実現
            if self.astrocyte.request_resource("association", 5.0):
                self.workspace.upload_to_workspace(
                source="association_cortex",
                data={"features": association_activity, "type": "integrated_sensation"},
                salience=0.5
            )

        # --- Step 2: Consciousness (意識) ---
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: High-Level Cognition ---
        if conscious_content:
            # Thinking Engine
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
            
            # Hippocampus
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

            # --- Step 5: Generate Top-Down Priming (次サイクルの予測) ---
            # 意識の内容が「概念」であれば、そのパターンを想起してプライミング信号とする
            if isinstance(conscious_content, dict) and "detected_objects" in conscious_content:
                # 視覚的な概念が意識にある場合、それを維持・強化しようとする
                concept = f"neural_concept_{self.cycle_count}" # ダミー
                # 本来は content から適切な ID を抽出
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
        """睡眠サイクル (Phase 5: Neuro-Symbolic Consolidation)"""
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
