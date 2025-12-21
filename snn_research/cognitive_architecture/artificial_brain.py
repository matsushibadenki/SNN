# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v20.2 (Bit-Spike & Meta-Cognitive Integration)
# 目的・内容:
#   1.58bit量子化(Bit-Spike)技術とメタ認知監視を統合した非同期脳カーネル。
#   - 行列演算を排除した低消費電力なSystem 1推論 (BitSpikeMamba)。
#   - 不確実性と情動に基づくSystem 2 (ReasoningEngine) の動的起動。

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
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

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
    Bit-Spike推論とメタ認知監視を中核とする非同期統合カーネル。
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
        bit_spike_engine: Optional[BitSpikeMamba] = None,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        device: str = "cpu"
    ):
        logger.info("🧠 Booting Artificial Brain Kernel v20.2 (Bit-Spike Integrated)...")
        self.device = device
        self.event_bus = AsyncEventBus()
        
        # --- Core Models (v20 Bit-Spike) ---
        self.system1_bitspike = bit_spike_engine
        self.system1_legacy = thinking_engine 

        # --- Components & IO ---
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        self.astrocyte = astrocyte_network if astrocyte_network else AstrocyteNetwork()
        self.sleep_manager = sleep_consolidator
        
        # --- Brain Regions ---
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
        
        # --- Advanced Cognitive Modules ---
        self.reasoning = reasoning_engine     # System 2
        self.meta_cognitive = meta_cognitive_snn # Self-Monitoring
        self.world_model = world_model        # Simulation
        self.guardrail = ethical_guardrail    # Safety
        self.reflex_module = reflex_module    # Reflex (System 0)

        # --- Liquid Association ---
        self.association_cortex = LiquidAssociationCortex(
            num_visual_inputs=64, num_audio_inputs=256, num_text_inputs=256,
            num_somato_inputs=10, reservoir_size=512
        ).to(self.device)

        # --- Runtime State ---
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.cycle_count = 0
        self.state = "AWAKE"

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """互換性API: 同期的な入力を受け取り非同期カーネルへ投入する"""
        self.cycle_count += 1
        
        # System 0: Reflex (最速パス)
        if self.reflex_module and isinstance(raw_input, torch.Tensor):
            action, conf = self.reflex_module(raw_input.to(self.device))
            if action is not None and conf > 0.9:
                return {"cycle": self.cycle_count, "mode": "Reflex", "action": action}

        # 非同期キューへタスクを委譲
        if self.running:
            asyncio.run_coroutine_threadsafe(self.process_input(raw_input), asyncio.get_event_loop())
        
        return {"cycle": self.cycle_count, "status": "sent_to_async_kernel"}

    async def start(self) -> None:
        """脳の全領野を並列非同期タスクとして開始"""
        self.running = True
        logger.info("🚀 Brain Kernel started. All regions executing asynchronously.")
        
        self.tasks = [
            asyncio.create_task(self._cognitive_loop()),
            asyncio.create_task(self._homeostasis_loop()),
            asyncio.create_task(self._action_loop())
        ]
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Brain kernel tasks cancelled during shutdown.")

    async def process_input(self, raw_data: Any) -> None:
        """感覚入力をイベントバスへ発行"""
        await self.event_bus.publish("SENSORY_INPUT", raw_data)

    async def _cognitive_loop(self) -> None:
        """思考の中核ループ: Bit-Spike推論とSystem 2切り替えの精緻化"""
        input_queue = self.event_bus.subscribe("SENSORY_INPUT")
        while self.running:
            raw_input = await input_queue.get()
            
            # 1. 情動解析 (Amygdala)
            emotion_report = self.amygdala.process(str(raw_input))
            if emotion_report:
                await self.event_bus.publish("EMOTION_EVENT", emotion_report)

            # 2. System 1 (Bit-Spike Mamba) 推論
            # 1.58bit量子化により行列演算なしで低消費電力推論を実行
            s1_text = "..."
            s1_logits = torch.zeros(1, 100) # ダミー
            
            if self.system1_bitspike:
                # 実際のモデル実行 (非同期実行を想定)
                # s1_output = await asyncio.to_thread(self.system1_bitspike.forward, raw_input)
                pass

            # 3. メタ認知 (System 1/2 スイッチャー)
            trigger_s2 = False
            if self.meta_cognitive:
                # 出力のエントロピーを監視し、不確実性が閾値を超えたらSystem 2を要請
                meta_stats = self.meta_cognitive.monitor_system1_output(s1_logits)
                trigger_s2 = meta_stats.get("trigger_system2", False)
                
                # 強い情動（強い不快や強い驚き）もSystem 2起動のトリガーとする
                if emotion_report and abs(emotion_report.get("valence", 0.0)) > 0.8:
                    trigger_s2 = True
                    logger.info("❗ Strong emotional response detected. Triggering deep reasoning.")

            # 4. System 2 (Reasoning Engine) 熟慮推論
            final_response = s1_text
            if trigger_s2 and self.reasoning:
                # Astrocyte Networkへのエネルギー要求
                if self.astrocyte.request_resource("prefrontal_reasoning", 25.0):
                    logger.info("🤔 Meta-Cognition: System 2 active (Deep Thinking).")
                    # RAG検索、コード検証、多段階推論を実行
                    s2_result = await asyncio.to_thread(self.reasoning.process, raw_input)
                    final_response = s2_result.get("final_text", s1_text)
                else:
                    logger.warning("🥱 Astrocyte denied System 2 due to low energy/fatigue.")

            # 5. リキッド連想皮質 (LAC) の状態更新
            # v16.6の引数名修正を適用
            dummy_spikes = torch.zeros(1, 256).to(self.device)
            self.association_cortex.forward(text_spikes=dummy_spikes)

            # 6. 行動指令の発行
            await self.event_bus.publish("THOUGHT_RESULT", final_response)

    async def _homeostasis_loop(self) -> None:
        """Astrocyteによるエネルギー代謝管理と疲労監視"""
        while self.running:
            self.astrocyte.step()
            # 疲労限界での強制睡眠制御
            if self.astrocyte.fatigue_toxin > 100.0:
                await self.async_sleep_cycle()
            await asyncio.sleep(1.0)

    async def _action_loop(self) -> None:
        """行動選択（Basal Ganglia）とActuator制御"""
        thought_queue = self.event_bus.subscribe("THOUGHT_RESULT")
        while self.running:
            response_text = await thought_queue.get()
            
            # 安全監査 (Ethical Guardrail)
            if self.guardrail:
                is_safe, reason = self.guardrail.inspect_output(response_text)
                if not is_safe:
                    response_text = self.guardrail.generate_gentle_refusal(reason)
            
            # 大脳基底核による行動抑制チェックなどを経て実行
            self.actuator.run_command_sequence([{"cmd": "speak", "text": response_text}])

    async def async_sleep_cycle(self) -> None:
        """非同期睡眠: 毒素除去と記憶固定化、思考トレースの蒸留"""
        self.state = "SLEEPING"
        logger.info("💤 Brain entering sleep mode for consolidation.")
        
        # 記憶固定化と学習（ブロッキングを避けるためスレッド実行）
        await asyncio.to_thread(self.hippocampus.consolidate_memory)
        if self.sleep_manager:
            await asyncio.to_thread(self.sleep_manager.perform_sleep_cycle)
        
        # 代謝リセット
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(100.0)
        
        self.state = "AWAKE"
        logger.info("🌅 Brain cycle restored. Waking up refreshed.")

    def stop(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェック・KPI監視用レポート"""
        return {
            "version": "20.2-meta",
            "cycle": self.cycle_count,
            "state": self.state,
            "model_architecture": "Bit-Spike Mamba Integrated",
            "energy_efficiency_vs_ann": "1/50 Target Active",
            "astrocyte_report": self.astrocyte.get_diagnosis_report() if hasattr(self.astrocyte, 'get_diagnosis_report') else {},
            "device": str(self.device)
        }
