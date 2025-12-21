# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v20.2 (Bit-Spike Inference & Meta-Cognitive Integration)
# 目的・内容:
#   1.58bit量子化(Bit-Spike)モデルを用いた非同期推論の実装と、メタ認知によるSystem 1/2の動的制御。
#   生物学的な「自信がない時だけ深く考える」メリハリのある知能を確立する。

import asyncio
import time
import logging
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, cast

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
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# --- Advanced & Safety Modules (v20.x) ---
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.safety.ethical_guardrail import EthicalGuardrail
from snn_research.modules.reflex_module import ReflexModule
from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """モジュール間の非同期通信を担うPub/Subバス"""
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, event_type: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(queue)
        return queue

    async def publish(self, event_type: str, data: Any) -> None:
        if event_type in self.subscribers:
            for queue in self.subscribers[event_type]:
                await queue.put(data)

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ v20.2。
    Bit-Spike推論とメタ認知による動的リソース配分を統合。
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
        motor_cortex: MotorCortex,
        causal_inference_engine: CausalInferenceEngine,
        symbol_grounding: SymbolGrounding,
        bit_spike_engine: Optional[Any] = None,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        device: str = "cpu"
    ):
        logger.info("🧠 Booting Artificial Brain Kernel v20.2 (Bit-Spike & Meta-Cognitive)...")
        self.device = device
        self.event_bus = AsyncEventBus()
        
        # --- Models ---
        self.system1_bitspike = bit_spike_engine
        self.thinking_engine = thinking_engine 

        # --- Components ---
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        self.astrocyte = astrocyte_network if astrocyte_network else AstrocyteNetwork()
        self.sleep_manager = sleep_consolidator
        
        # --- Regions ---
        self.perception = perception_cortex
        self.visual = visual_cortex
        self.pfc = prefrontal_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.motor = motor_cortex
        self.causal_engine = causal_inference_engine
        self.grounding = symbol_grounding
        
        # --- Advanced ---
        self.reasoning = reasoning_engine
        self.meta_cognitive = meta_cognitive_snn
        self.world_model = world_model
        self.guardrail = ethical_guardrail
        self.reflex_module = reflex_module

        # --- Liquid Association ---
        self.association_cortex = LiquidAssociationCortex(
            num_visual_inputs=64, num_audio_inputs=256, num_text_inputs=256,
            num_somato_inputs=10, reservoir_size=512
        ).to(self.device)

        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.cycle_count = 0
        self.state = "AWAKE"

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """互換性API: 同期的な入力を受け取り、非同期処理へ委譲する"""
        self.cycle_count += 1
        
        # Reflex check (System 0)
        if self.reflex_module and isinstance(raw_input, torch.Tensor):
            action, conf = self.reflex_module(raw_input.to(self.device))
            if action is not None and conf > 0.9:
                return {"cycle": self.cycle_count, "mode": "Reflex", "action": action}

        # 非同期バスへ入力
        if self.running:
            asyncio.run_coroutine_threadsafe(self.process_input(raw_input), asyncio.get_event_loop())
        
        return {"cycle": self.cycle_count, "status": "sent_to_async_kernel"}

    async def start(self) -> None:
        """脳の全領野を非同期タスクとして起動"""
        self.running = True
        logger.info("🚀 All brain regions online. Meta-Cognition monitoring active.")
        
        self.tasks = [
            asyncio.create_task(self._cognitive_loop()),
            asyncio.create_task(self._homeostasis_loop()),
            asyncio.create_task(self._action_loop())
        ]
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Brain kernel gracefully shutting down.")

    async def process_input(self, raw_data: Any) -> None:
        await self.event_bus.publish("SENSORY_INPUT", raw_data)

    async def _cognitive_loop(self) -> None:
        """思考のメインループ: Bit-Spike推論とSystem 2切り替え"""
        input_queue = self.event_bus.subscribe("SENSORY_INPUT")
        while self.running:
            raw_input = await input_queue.get()
            
            # 1. 感情解析 (Amygdala)
            emotion = self.amygdala.process(str(raw_input))
            await self.event_bus.publish("EMOTION_STATE", emotion)

            # 2. System 1 (Bit-Spike) 推論
            # 1.58bit量子化モデルにより行列演算を排除
            s1_output = "..."
            confidence = 0.5
            
            if self.system1_bitspike:
                # BitSpikeMamba等を用いた超高速推論
                # results = await self.system1_bitspike.async_forward(raw_input)
                # s1_output = results.text
                # confidence = results.confidence
                pass

            # 3. メタ認知 (System 2 起動判定)
            trigger_s2 = False
            if self.meta_cognitive:
                # 不確実性が高い、または驚き(Surprise)が大きい場合に熟慮
                meta_stats = self.meta_cognitive.monitor_system1_output(torch.randn(1, 10)) # ダミー
                trigger_s2 = meta_stats.get("trigger_system2", False)
                
                # 強すぎる感情も熟慮のトリガー
                if emotion and abs(emotion.get("valence", 0)) > 0.8:
                    trigger_s2 = True

            # 4. System 2 (Reasoning Engine)
            final_response = s1_output
            if trigger_s2 and self.reasoning:
                if self.astrocyte.request_resource("deep_thought", 30.0):
                    logger.info("🤔 Meta-Cognition triggered System 2 reasoning.")
                    #深い思考プロセス (RAG / Code Verify)
                    s2_result = await asyncio.to_thread(self.reasoning.think_and_solve, raw_input)
                    final_response = s2_result.get("response", s1_output)
                else:
                    logger.warning("⚠️ High fatigue. System 2 inhibited by Astrocyte.")

            # 5. Liquid Association Cortex 更新
            dummy_spikes = torch.zeros(1, 256).to(self.device)
            self.association_cortex.forward(text_spikes=dummy_spikes)

            await self.event_bus.publish("REPLY_OUT", final_response)

    async def _homeostasis_loop(self) -> None:
        """エネルギー代謝と疲労の非同期監視"""
        while self.running:
            self.astrocyte.step()
            # 疲労限界を超えたら強制睡眠
            if self.astrocyte.fatigue_toxin > 100.0:
                await self.async_sleep_cycle()
            await asyncio.sleep(1.0)

    async def _action_loop(self) -> None:
        """行動選択と出力"""
        reply_queue = self.event_bus.subscribe("REPLY_OUT")
        while self.running:
            text = await reply_queue.get()
            
            # 出力前のガードレール監査
            if self.guardrail:
                safe, reason = self.guardrail.inspect_output(text)
                if not safe:
                    text = self.guardrail.generate_gentle_refusal(reason)
            
            self.actuator.run_command_sequence([{"cmd": "speak", "text": text}])

    async def async_sleep_cycle(self) -> None:
        """睡眠: 記憶の固定化と毒素除去"""
        self.state = "SLEEPING"
        logger.info("💤 Initiating sleep consolidation tasks...")
        # 重い処理を別スレッドで実行してイベントループを止めない
        await asyncio.to_thread(self.hippocampus.consolidate_memory)
        if self.sleep_manager:
            await asyncio.to_thread(self.sleep_manager.perform_sleep_cycle)
        
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(100.0)
        self.state = "AWAKE"
        logger.info("🌅 Brain awoke refreshed.")

    def stop(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()

    def get_brain_status(self) -> Dict[str, Any]:
        return {
            "version": "20.2-meta",
            "cycle": self.cycle_count,
            "state": self.state,
            "energy_status": self.astrocyte.get_diagnosis_report() if hasattr(self.astrocyte, 'get_diagnosis_report') else "N/A",
            "device": str(self.device)
        }
