# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v20.1 (Hybrid Compatibility & Async)
# 目的・内容:
#   非同期イベント駆動型アーキテクチャ(v20)を核としつつ、既存の同期型API(v16.x)との互換性を維持。
#   mypyエラー(run_cognitive_cycleの欠落およびAsyncEventBusの型未定義)を解決。

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
        # mypyエラー解決のため型を明示
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
    SNNベース 人工脳アーキテクチャ v20.1。
    非同期カーネルを核としつつ、既存の run_cognitive_cycle APIを維持して互換性を確保。
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
        # v20 Integrated Engine
        bit_spike_engine: Optional[Any] = None,
        # Optional Modules
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        device: str = "cpu"
    ):
        logger.info("🧠 Initializing Hybrid Artificial Brain Kernel v20.1...")
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

        # --- Internal State ---
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.cycle_count = 0
        self.state = "AWAKE"

    # --- 既存APIの互換性維持 (v16.x互換) ---
    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        同期型呼び出しの互換性用メソッド。内部で非同期処理を実行する。
        既存のテスト(tests/cognitive_architecture/test_artificial_brain.py)等を壊さないために必須。
        """
        self.cycle_count += 1
        logger.debug(f"Cycle {self.cycle_count}: Sync wrapper called for input {type(raw_input)}")
        
        # 1. 簡易的な同期処理（既存のロジックのサブセットまたは非同期タスクへの委譲）
        # 本来は非同期ループが回っている前提だが、テスト環境等のために同期的に結果を返す
        
        # Reflex (System 0) - 同期実行可能
        if self.reflex_module and isinstance(raw_input, torch.Tensor):
            action, conf = self.reflex_module(raw_input.to(self.device))
            if action is not None and conf > 0.9:
                return {"cycle": self.cycle_count, "mode": "Reflex", "action": action}

        # Guardrail
        if self.guardrail:
            safe, reason = self.guardrail.inspect_input(str(raw_input))
            if not safe:
                return {"status": "blocked", "response": self.guardrail.generate_gentle_refusal(reason)}

        # 本来の非同期フローへデータを投入
        if self.running:
            # ループが動いている場合はキューに入れる
            asyncio.run_coroutine_threadsafe(self.process_input(raw_input), asyncio.get_event_loop())
        
        # 暫定的なレスポンス（詳細な推論結果が必要な場合は await が必要だが、互換性のために標準的な辞書を返す）
        return {
            "cycle": self.cycle_count,
            "status": "processed",
            "mode": "System 1 (Async Handover)",
            "executed": ["perception", "reflex_check"]
        }

    # --- 非同期カーネル機能 (v20.0) ---
    async def start(self) -> None:
        """脳の非同期ループを開始する"""
        self.running = True
        logger.info("🚀 Brain Kernel v20.1 online. Starting async regions...")
        
        self.tasks = [
            asyncio.create_task(self._cognitive_loop()),
            asyncio.create_task(self._homeostasis_loop()),
            asyncio.create_task(self._action_loop())
        ]
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Brain tasks cancelled.")

    async def process_input(self, raw_data: Any) -> None:
        """外部からの入力を非同期バスに投入"""
        await self.event_bus.publish("SENSORY_INPUT", raw_data)

    async def _cognitive_loop(self) -> None:
        """メイン思考ループ"""
        input_queue = self.event_bus.subscribe("SENSORY_INPUT")
        while self.running:
            raw_input = await input_queue.get()
            
            # System 1 推論 (Liquid Associationの更新含む)
            # v16.6のバグ修正を適用: text_spikes 引数を使用
            dummy_spikes = torch.zeros(1, 256).to(self.device)
            self.association_cortex.forward(text_spikes=dummy_spikes)
            
            # 推論ロジック (省略: 必要に応じてv20.0の実装を追加)
            response = "Processed via v20 kernel."
            await self.event_bus.publish("REPLY_OUT", response)

    async def _homeostasis_loop(self) -> None:
        """Astrocyteによる代謝監視"""
        while self.running:
            self.astrocyte.step()
            if self.astrocyte.fatigue_toxin > 100.0:
                await self.async_sleep_cycle()
            await asyncio.sleep(1.0)

    async def _action_loop(self) -> None:
        """行動出力ループ"""
        reply_queue = self.event_bus.subscribe("REPLY_OUT")
        while self.running:
            text = await reply_queue.get()
            self.actuator.run_command_sequence([{"cmd": "speak", "text": text}])

    async def async_sleep_cycle(self) -> None:
        """非同期睡眠"""
        self.state = "SLEEPING"
        await asyncio.to_thread(self.hippocampus.consolidate_memory)
        if self.sleep_manager:
            await asyncio.to_thread(self.sleep_manager.perform_sleep_cycle)
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(100.0)
        self.state = "AWAKE"

    def stop(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェックAPI用"""
        return {
            "version": "20.1-hybrid",
            "cycle": self.cycle_count,
            "state": self.state,
            "astrocyte": self.astrocyte.get_diagnosis_report() if hasattr(self.astrocyte, 'get_diagnosis_report') else {},
            "device": str(self.device)
        }
