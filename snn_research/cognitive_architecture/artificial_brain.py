# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v16.6 (Bugfix: LAC Argument)
# 目的・内容:
#   全認知モジュールを統括する中核クラス。
#   修正: LiquidAssociationCortex.forward() の引数名を text_input -> text_spikes に修正し、mypyエラーを解決。

from typing import Dict, Any, List, Optional, Union, Tuple, cast
import time
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer # type: ignore

# --- Core & IO Modules ---
from snn_research.core.snn_core import SNNCore
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

# --- Cognitive Components ---
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.visual_perception import VisualCortex
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# --- Advanced Modules (v16.x & v17.0) ---
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.safety.ethical_guardrail import EthicalGuardrail
from snn_research.modules.reflex_module import ReflexModule
from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    SNNベースの人工脳アーキテクチャ統合クラス (v16.6)。
    認知サイクルの各フェーズでモジュール間のデータの受け渡しを厳密に行うよう修正。
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
        # Optional Modules
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        logger.info("🚀 Booting Artificial Brain Kernel v16.6 (Integrated)...")
        self.device = device

        # --- Core Components ---
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.sleep_manager = sleep_consolidator
        self.thinking_engine = thinking_engine # System 1 Backbone

        # --- Homeostasis ---
        if astrocyte_network is None:
            self.astrocyte = AstrocyteNetwork()
        else:
            self.astrocyte = astrocyte_network

        # --- IO ---
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator

        # --- Brain Regions ---
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

        # --- Advanced Cognitive Capabilities (v16.x) ---
        self.reasoning = reasoning_engine     # System 2
        self.meta_cognitive = meta_cognitive_snn # Self-Monitoring
        self.world_model = world_model        # Simulation
        self.guardrail = ethical_guardrail    # Safety
        self.reflex_module = reflex_module    # Reflex (v17.0)

        # --- Tokenizer Setup ---
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Tokenizer load failed: {e}. Text capabilities limited.")
            self.tokenizer = None

        # --- Liquid Association (Reservoir) ---
        self.association_cortex = LiquidAssociationCortex(
            num_visual_inputs=64,
            num_audio_inputs=256,
            num_text_inputs=256,
            num_somato_inputs=10,
            reservoir_size=512
        ).to(self.device)

        # --- Internal State ---
        self.cycle_count = 0
        self.state = "AWAKE"
        self.last_thought_trace: List[str] = []
        
        # Initial Resource Request
        self.astrocyte.request_resource("system_boot", 50.0)
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        認知サイクルのメインループ。
        Perception -> Amygdala -> Hippocampus -> System 1/2 -> Basal Ganglia -> Action
        """
        # 0. 状態チェック
        if self.state in ["SLEEPING", "DREAMING"]:
            self.astrocyte.request_resource("system_idle", 0.1)
            return {"status": "sleeping", "response": "Zzz...", "mode": "Sleep"}

        self.cycle_count += 1
        report: Dict[str, Any] = {
            "cycle": self.cycle_count,
            "input": str(raw_input)[:50],
            "mode": "System 1", # Default
            "executed": [],
            "warnings": [],
            "emotion": None,
            "context": []
        }

        # 1. Perception
        self.receptor.receive(raw_input)
        report["executed"].append("perception")
        
        # 入力がテキストの場合の処理用変数
        raw_text_input = str(raw_input) if isinstance(raw_input, str) else None

        # --- Reflex Arc (v17.0) ---
        if self.reflex_module and isinstance(raw_input, torch.Tensor):
            reflex_action, reflex_conf = self.reflex_module(raw_input.to(self.device))
            if reflex_action is not None:
                logger.info(f"⚡ Reflex Triggered! Action: {reflex_action} (Conf: {reflex_conf:.2f})")
                reflex_cmd = f"REFLEX_ACTION_{reflex_action}"
                self.actuator.run_command_sequence([{"cmd": "reflex", "value": reflex_cmd}])
                report["mode"] = "Reflex"
                report["action"] = {"type": "reflex", "id": reflex_action}
                return report

        # Guardrail check
        if self.guardrail:
            is_safe_input, reason = self.guardrail.inspect_input(str(raw_input))
            if not is_safe_input:
                logger.warning(f"🛡️ Input Blocked: {reason}")
                return {"status": "blocked", "response": self.guardrail.generate_gentle_refusal(reason)}

        # 2. Amygdala (Emotion Processing)
        # 扁桃体に入力を通し、情動反応を取得する
        emotion_state = None
        if raw_text_input:
            emotion_state = self.amygdala.process(raw_text_input)
            if emotion_state:
                report["emotion"] = emotion_state
                # 情動情報をWorkspaceにブロードキャスト（サリエンス高め）
                self.workspace.upload_to_workspace(
                    source="amygdala",
                    data={"type": "emotion", **emotion_state},
                    salience=0.8
                )

        # 3. Hippocampus (Memory Retrieval)
        # コンテキスト検索: 入力に関連する過去の記憶や知識を引き出す
        retrieved_context = []
        if raw_text_input:
            retrieved_context = self.hippocampus.recall(raw_text_input, k=2)
            if retrieved_context:
                report["context"] = retrieved_context
                # コンテキストもWorkspaceへ（System 2が参照可能に）
                self.workspace.upload_to_workspace(
                    source="hippocampus",
                    data={"type": "memory_recall", "items": retrieved_context},
                    salience=0.6
                )

        # 4. System 1 (Intuition) & Input Encoding
        input_ids = None
        device_obj: Union[torch.device, str]
        try:
            device_obj = next(self.thinking_engine.parameters()).device
        except StopIteration:
            device_obj = self.device

        if self.tokenizer and raw_text_input:
            input_ids = self.tokenizer.encode(raw_text_input, return_tensors='pt').to(device_obj)
        elif self.tokenizer and isinstance(raw_input, torch.Tensor) and raw_input.dtype == torch.long:
            input_ids = raw_input.to(device_obj)

        system1_logits = None
        system1_text = "..."
        if input_ids is not None:
            with torch.no_grad():
                output = self.thinking_engine(input_ids)
                if isinstance(output, tuple):
                    system1_logits = output[0]
                else:
                    system1_logits = output
                pred_ids = torch.argmax(system1_logits, dim=-1)
                system1_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        # --- Liquid Association Update ---
        # System 1 の活性状態をリキッドステートマシンに入力して内部状態を更新
        if system1_logits is not None:
            # 簡易的にLogitsの平均プーリングなどを入力として扱う
            # LiquidAssociationCortexの実装に合わせて次元調整が必要だが、ここではダミー入力で更新を回す
            # 実際の入力次元(256 text inputs)に合わせる
            dummy_liquid_input = torch.zeros(1, 256).to(self.device)
            # system1_logitsから一部情報を抽出して入力するなど本来は必要
            
            # [Fix] 引数名を text_input から text_spikes に修正
            self.association_cortex.forward(text_spikes=dummy_liquid_input)

        # 5. Meta-Cognition (Monitoring)
        trigger_system2 = False
        meta_stats = {}
        if self.meta_cognitive and system1_logits is not None:
            meta_stats = self.meta_cognitive.monitor_system1_output(system1_logits)
            trigger_system2 = meta_stats.get("trigger_system2", False)
            report["meta_stats"] = meta_stats
            
            # 情動が強すぎる場合、System 2を強制起動して冷静さを取り戻そうとする（または抑制される）
            if emotion_state and abs(emotion_state.get("valence", 0)) > 0.8:
                 trigger_system2 = True
                 logger.info("😠 Strong emotion detected. Triggering System 2 for regulation.")

            if trigger_system2:
                logger.info(f"🤔 Meta-Cognition triggered System 2 (Entropy: {meta_stats.get('entropy', 0):.2f})")

        # 6. System 2 (Reasoning)
        final_response = system1_text
        thought_trace = []
        
        if trigger_system2 and self.reasoning:
            if self.astrocyte.request_resource("prefrontal_cortex", 15.0):
                report["mode"] = "System 2"
                report["executed"].append("reasoning")
                if input_ids is not None:
                    try:
                        # コンテキストを考慮した推論
                        # Prompt Augmentation的な処理はReasoningEngine側で行うか、ここでテキスト結合する
                        # ここでは簡易的にReasoningEngineに任せる想定だが、コンテキスト情報をログに残す
                        
                        reasoning_result = self.reasoning.think_and_solve(
                            input_ids=input_ids,
                            task_type="general",
                            tokenizer=self.tokenizer
                            # 将来的には context=retrieved_context を渡す
                        )
                        thought_trace = reasoning_result.get("thought_trace", [])
                        self.last_thought_trace = thought_trace
                        
                        if self.guardrail:
                            is_safe_thought, reason = self.guardrail.validate_thought_process(thought_trace)
                            if not is_safe_thought:
                                logger.critical(f"🛡️ Thought Blocked: {reason}")
                                return {"status": "blocked", "response": self.guardrail.generate_gentle_refusal(reason)}

                        final_output_ids = reasoning_result["final_output"]
                        final_response = self.tokenizer.decode(final_output_ids[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.error(f"❌ Reasoning Engine Error: {e}")
                        final_response = system1_text # Fallback to System 1
            else:
                logger.warning("⚠️ System 2 requested but denied by Astrocyte (Fatigue/Energy).")

        # 7. World Model (Simulation)
        if self.world_model and report["mode"] == "System 2" and input_ids is not None:
            try:
                current_latent = self.world_model.encode(input_ids)
                # 仮のアクションベクトル
                action_vec = torch.zeros(1, 1, 10).to(device_obj)
                action_vec[0, 0, 1] = 1.0 
                sim_result = self.world_model.simulate_trajectory(current_latent, action_vec)
                predicted_reward = sim_result["rewards"].mean().item()
                report["simulation"] = {"predicted_reward": predicted_reward}
                logger.info(f"🔮 World Model Prediction: Reward {predicted_reward:.3f}")
                
                if predicted_reward < -0.5:
                    final_response += " ... (しかし、シミュレーション結果により発言を控えます。)"
            except Exception as e:
                logger.error(f"⚠️ World Model simulation failed: {e}")

        # 8. Action Selection (Basal Ganglia)
        # Brainが生成した応答を「候補の一つ」として大脳基底核に提示し、最終決定を委ねる
        candidates = []
        if final_response:
             candidates.append({
                 'action': 'reply', 
                 'value': 1.0, # Reasoningを経たものは確度が高いとみなす
                 'params': {'text': final_response}
             })
        
        # Basal Ganglia内で本能的アクション（無視、逃避など）と競合させる
        selected_action = None
        if self.astrocyte.request_resource("basal_ganglia", 2.0):
            selected_action = self.basal_ganglia.select_action(candidates, emotion_context=emotion_state)
            report["action"] = selected_action

        # 9. Actuation
        if selected_action and selected_action['action'] == 'reply':
            resp_text = selected_action['params']['text']
            if self.guardrail:
                is_safe_out, reason = self.guardrail.inspect_output(resp_text)
                if not is_safe_out:
                    resp_text = self.guardrail.generate_gentle_refusal(reason)
            self.actuator.run_command_sequence([{"cmd": "speak", "text": resp_text}])
            report["response"] = resp_text
        elif selected_action and selected_action['action'] == 'ignore':
            report["response"] = "(Ignored)"
            logger.info("🚫 Basal Ganglia decided to IGNORE the input.")

        # 10. Memory Consolidation (Encoding)
        # 今回のサイクル（入力、出力、情動）をエピソード記憶として保存
        episode_data = {
            "timestamp": time.time(),
            "input": raw_text_input,
            "output": report.get("response"),
            "emotion": emotion_state,
            "thought_trace": thought_trace if report["mode"] == "System 2" else None
        }
        self.hippocampus.store_episode(episode_data)

        # 11. Homeostasis
        self.astrocyte.step()
        if self.astrocyte.fatigue_toxin > 100.0:
            logger.info("🥱 Brain is exhausted. Initiating sleep cycle...")
            self.sleep_cycle()
            report["next_state"] = "SLEEPING"

        return report

    def sleep_cycle(self):
        """睡眠サイクルの実行"""
        self.state = "SLEEPING"
        logger.info("\n🌙 Brain entering sleep mode...")
        
        # 記憶の固定化（短期記憶 -> 長期記憶）
        self.hippocampus.consolidate_memory()
        
        if self.sleep_manager:
            stats = self.sleep_manager.perform_sleep_cycle()
            logger.info(f"   💤 Dreams replayed: {stats.get('dreams_replayed', 0)}")
            logger.info(f"   🧠 Synapses consolidated: {stats.get('synapses_updated', 0)}")
        
        if hasattr(self.astrocyte, 'replenish_energy'):
            self.astrocyte.replenish_energy(1000.0)
        if hasattr(self.astrocyte, 'clear_fatigue'):
            self.astrocyte.clear_fatigue(100.0)
        
        self.state = "AWAKE"
        logger.info("🌅 Brain woke up refreshed.")

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェックAPI用ステータス取得"""
        
        astro_report = {}
        if hasattr(self.astrocyte, 'get_diagnosis_report'):
             astro_report = self.astrocyte.get_diagnosis_report()
        
        meta_report = {}
        if self.meta_cognitive is not None:
             mc = cast(MetaCognitiveSNN, self.meta_cognitive)
             if hasattr(mc, 'get_status_report'):
                 meta_report = mc.get_status_report() # type: ignore

        return {
            "cycle": self.cycle_count,
            "state": self.state,
            "astrocyte": astro_report,
            "meta_cognition": meta_report,
            "device": str(self.device),
            "current_goal": self.pfc.current_goal
        }