# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v16.1 (Syntax Fix)
# Description:
#   ROADMAP v16 対応の統合人工脳カーネル。
#   修正: 末尾の不要な閉じ括弧を削除し、SyntaxErrorを解消。
#         LiquidAssociationCortexの入力次元対応 (num_audio_inputs=256) を維持。

from typing import Dict, Any, List, Optional, Union, cast
import time
import logging
import torch
from torchvision import transforms  # type: ignore
from transformers import AutoTokenizer # type: ignore

# Core Modules
from snn_research.core.snn_core import SNNCore
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

# --- Core Networks ---
from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex

# --- Cognitive Modules ---
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

# --- New Modules for v16 ---
from .reasoning_engine import ReasoningEngine
from snn_research.safety.ethical_guardrail import EthicalGuardrail

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    Artificial Brain Kernel v16.0
    System 1 (直感) と System 2 (熟慮) を統合し、倫理的なガードレールで保護された人工脳。
    """
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        motivation_system: IntrinsicMotivationSystem,
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        thinking_engine: SNNCore, # Base generative model (SFormer)
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
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        tokenizer_name: str = "gpt2"
    ):
        logger.info("🚀 Booting Artificial Brain Kernel v16.0 (Ethical AGI Prototype)...")
        
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        self.thinking_engine = thinking_engine # System 1 Model
        
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
        
        # --- System 2 & Safety ---
        self.reasoning = reasoning_engine
        self.guardrail = ethical_guardrail
        
        # トークナイザ (思考エンジン用)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Tokenizer load failed: {e}. Text reasoning will be limited.")
            self.tokenizer = None

        # --- Liquid Association Cortex (Unified Perception) ---
        # 汎用入力(256dim)に対応するため num_audio_inputs を拡張
        self.association_cortex = LiquidAssociationCortex(
            num_visual_inputs=64,
            num_audio_inputs=256, 
            num_text_inputs=256,
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
        
        # 初期リソース確保
        self.astrocyte.request_resource("thinking_engine", 50.0) 
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する。
        Observation -> Safety Check -> Perception -> Awareness -> Reasoning -> Action
        """
        # 0. 状態チェック
        if self.state in ["SLEEPING", "DREAMING"]:
            self.astrocyte.request_resource("system_idle", 0.1)
            return {"status": "sleeping", "response": "Zzz..."}

        self.cycle_count += 1
        report: Dict[str, Any] = {
            "cycle": self.cycle_count,
            "input_preview": str(raw_input)[:50],
            "executed_modules": [],
            "denied_modules": [],
            "alerts": []
        }

        # --- Step 0: Input Guardrail (安全検査) ---
        if self.guardrail:
            # 入力がテキストの場合のチェック
            input_text = str(raw_input) # 簡易変換
            is_safe, reason = self.guardrail.inspect_input(input_text)
            if not is_safe:
                logger.warning(f"🛡️ Input blocked by guardrail: {reason}")
                refusal_msg = self.guardrail.generate_gentle_refusal(reason)
                return {"status": "blocked", "response": refusal_msg, "reason": reason}

        # --- Step 1: Perception (Unified & Modal) ---
        sensory_info = self.receptor.receive(raw_input)
        
        visual_spikes_for_lac: Optional[torch.Tensor] = None
        audio_spikes_for_lac: Optional[torch.Tensor] = None
        
        # ... (既存の知覚プロセス) ...
        if sensory_info['type'] == 'image':
            if self.astrocyte.request_resource("visual_cortex", 15.0):
                img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
                self.visual.perceive_and_upload(img_tensor)
                report["executed_modules"].append("visual_cortex")
                visual_spikes_for_lac = torch.rand(1, 64) > 0.9 # Dummy
                
                if self.astrocyte.request_resource("symbol_grounding", 5.0):
                    vis_data = self.workspace.get_information("visual_cortex")
                    if vis_data and "features" in vis_data:
                        self.grounding.ground_neural_pattern(vis_data["features"], "visual_input")
                        report["executed_modules"].append("symbol_grounding")
            else:
                report["denied_modules"].append("visual_cortex")
        else:
            if self.astrocyte.request_resource("perception", 2.0):
                spike_pattern = self.encoder.encode(sensory_info, duration=16)
                self.perception.perceive_and_upload(spike_pattern)
                report["executed_modules"].append("perception")
                # LACのAudio入力(256)にマップ
                audio_spikes_for_lac = spike_pattern.float().mean(dim=0).unsqueeze(0) > 0.5
                
                if self.astrocyte.request_resource("amygdala", 1.0):
                    self.amygdala.evaluate_and_upload(str(sensory_info['content']))
                    report["executed_modules"].append("amygdala")
            else:
                report["denied_modules"].append("perception")

        # Liquid Association Cortex
        lac_vis = visual_spikes_for_lac.float() if visual_spikes_for_lac is not None else None
        lac_aud = audio_spikes_for_lac.float() if audio_spikes_for_lac is not None else None
        
        # デバイスの整合性を確保
        device = next(self.association_cortex.parameters()).device
        if lac_vis is not None: lac_vis = lac_vis.to(device)
        if lac_aud is not None: lac_aud = lac_aud.to(device)

        association_activity = self.association_cortex(visual_spikes=lac_vis, audio_spikes=lac_aud)
        
        if self.astrocyte.request_resource("association", 5.0):
            self.workspace.upload_to_workspace(
                source="association_cortex",
                data={"features": association_activity, "type": "integrated_sensation"},
                salience=0.3 
            )
            report["executed_modules"].append("association_cortex")

        # --- Step 2: Consciousness (意識の放送) ---
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: High-Level Cognition (Thinking & Verifier) ---
        thought_output = None
        
        if conscious_content:
            # PFC & Causal Engine Update
            if self.astrocyte.request_resource("prefrontal_cortex", 8.0):
                self.pfc.handle_conscious_broadcast("workspace", conscious_content)
                report["executed_modules"].append("prefrontal_cortex")

            # --- Reasoning Engine (System 2 Loop) ---
            # テキストベースの思考が可能な場合
            if (self.reasoning and self.tokenizer and 
                (isinstance(conscious_content, str) or isinstance(sensory_info.get('content'), str))):
                
                target_text = str(conscious_content) if isinstance(conscious_content, str) else str(sensory_info.get('content'))
                
                try:
                    # デバイス特定
                    device = next(cast(torch.nn.Module, self.thinking_engine).parameters()).device
                    input_ids = self.tokenizer.encode(target_text, return_tensors='pt').to(device)
                    
                    # 思考実行 (Think & Solve)
                    think_result = self.reasoning.think_and_solve(
                        input_ids, 
                        task_type="general", # 将来的には分類器で決定
                        temperature=0.7
                    )
                    
                    # 思考トレースの監査 (Thought Audit)
                    if self.guardrail:
                        is_thought_safe, audit_msg = self.guardrail.validate_thought_process(think_result['thought_trace'])
                        if not is_thought_safe:
                            logger.warning(f"🛑 Unsafe thought detected: {audit_msg}")
                            report["alerts"].append(f"Thought blocked: {audit_msg}")
                            # 思考結果を破棄または修正
                            think_result["final_output"] = None 
                            thought_output = "[Thought Blocked by Safety Protocol]"
                        else:
                            # デコード
                            thought_output = self.tokenizer.decode(think_result['final_output'][0], skip_special_tokens=True)
                            report["thought_strategy"] = think_result['strategy']
                    else:
                        thought_output = self.tokenizer.decode(think_result['final_output'][0], skip_special_tokens=True)

                    if thought_output:
                        report["thought_process"] = thought_output
                        report["executed_modules"].append("reasoning_engine")
                        self.workspace.upload_to_workspace("reasoning_engine", thought_output, salience=0.7)
                        
                except Exception as e:
                    logger.error(f"Reasoning error: {e}")
            
            # Fallback to System 1 (Thinking Engine direct call) if Reasoning Engine failed or not available
            elif self.astrocyte.request_resource("thinking_engine", 20.0):
                try:
                    # Dummy or simple forward
                    device = next(self.thinking_engine.parameters()).device
                    dummy_ids = torch.randint(0, 1000, (1, 16)).to(device)
                    _ = self.thinking_engine(dummy_ids)
                    thought_output = "[System 1 Neural Activity]"
                    report["thought_process"] = thought_output
                    report["executed_modules"].append("thinking_engine (sys1)")
                except Exception:
                    pass

            # --- Action Selection ---
            amygdala_state = self.workspace.get_information("amygdala")
            if self.astrocyte.request_resource("basal_ganglia", 3.0):
                # 行動候補の生成（本来はPlanner等から来るがここでは簡易）
                candidates = [
                    {'action': 'reply_text', 'value': 0.8, 'params': {'text': thought_output}}, 
                    {'action': 'store_memory', 'value': 0.6},
                    {'action': 'ignore', 'value': 0.1}
                ]
                
                selected_action = self.basal_ganglia.select_action(candidates, emotion_context=amygdala_state)
                report["executed_modules"].append("basal_ganglia")
                
                if selected_action:
                    # --- Output Guardrail (Action & Output Check) ---
                    action_allowed = True
                    
                    if self.guardrail:
                        # アクション自体のチェック
                        is_safe_act, act_reason = self.guardrail.validate_action(selected_action)
                        if not is_safe_act:
                            action_allowed = False
                            report["alerts"].append(f"Action blocked: {act_reason}")
                        
                        # 出力テキストのチェック
                        if action_allowed and selected_action.get('action') == 'reply_text':
                            text_content = selected_action.get('params', {}).get('text', "")
                            is_safe_out, out_reason = self.guardrail.inspect_output(text_content)
                            if not is_safe_out:
                                action_allowed = False
                                report["alerts"].append(f"Output text blocked: {out_reason}")
                                # 拒否メッセージに差し替え
                                selected_action['params']['text'] = self.guardrail.generate_gentle_refusal(out_reason)
                                action_allowed = True # 差し替えたので許可

                    if action_allowed:
                        action_name = selected_action.get('action')
                        report["action"] = action_name
                        
                        # Motor Execution
                        if self.astrocyte.request_resource("motor_cortex", 10.0):
                            motor_commands = self.cerebellum.refine_action_plan(selected_action)
                            execution_log = self.motor.execute_commands(motor_commands)
                            self.actuator.run_command_sequence(execution_log)
                            report["executed_modules"].append("motor_cortex")
            
            # Memory Storage
            if self.astrocyte.request_resource("hippocampus", 4.0):
                episode = {
                    "timestamp": time.time(),
                    "input": str(raw_input),
                    "consciousness": conscious_content,
                    "thought": thought_output,
                    "action": selected_action if 'selected_action' in locals() else None,
                    "emotion": amygdala_state
                }
                self.hippocampus.store_episode(episode)
                report["executed_modules"].append("hippocampus")

            # Priming
            if isinstance(conscious_content, dict) and "detected_objects" in conscious_content:
                concept = f"neural_concept_{self.cycle_count}"
                priming = self.grounding.recall_pattern(concept)
                if priming is not None:
                    self.current_priming_signal = priming
            else:
                self.current_priming_signal = None

        # --- Step 4: System Check (Homeostasis) ---
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
        """睡眠と夢（記憶の整理）"""
        self.state = "SLEEPING"
        print("\n🌙 --- SLEEP CYCLE INITIATED ---")
        
        phases = []
        syn_change_total = 0.0
        
        # 1. Hippocampus -> Cortex Consolidation
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        if episodes:
            print(f"   📝 Consolidating {len(episodes)} episodes...")
            for ep in episodes:
                self.cortex.consolidate_memory(ep)
            phases.append(f"Consolidated {len(episodes)} episodes")
        
        # 2. Generative Replay (Dreaming)
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
        
        # 3. Energy Replenishment
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
        """外部からの知識修正を受け付ける"""
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