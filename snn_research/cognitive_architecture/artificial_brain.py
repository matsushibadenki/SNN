# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v21.0 (Advanced Cognitive Integration)
# 目的・内容:
#   ロードマップ v21.0 に向けた、System 1/2 の統合、動的リソース制御、
#   およびグローバルワークスペースを介した意識的推論の実装。
#   - MetaCognitiveSNN による不確実性（Surprise）監視と System 2 への自動委譲。
#   - Astrocyte Network と連動したスパイクバースト抑制（ホメオスタシス）。
#   - 非同期イベントバスによる知覚・思考・行動の完全な疎結合化。

import asyncio
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple, cast

# --- Core & IO ---
from snn_research.core.snn_core import SNNCore
from snn_research.io.actuator import Actuator

# --- Cognitive Components ---
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """モジュール間の非同期通信を担う優先度付きPub/Subバス"""
    def __init__(self) -> None:
        # 優先度付きキューを使用（緊急の反射や倫理アラートを優先）
        self.subscribers: Dict[str, List[asyncio.PriorityQueue]] = {}

    def subscribe(self, event_type: str) -> asyncio.PriorityQueue:
        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(queue)
        return queue

    async def publish(self, event_type: str, data: Any, priority: int = 10) -> None:
        if event_type in self.subscribers:
            for queue in self.subscribers[event_type]:
                # priorityが低いほど優先順位が高い
                await queue.put((priority, data))

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ v21.0。
    生物学的ホメオスタシスと高度な推論戦略を統合したプロフェッショナル版。
    """
    def __init__(
        self,
        config: Dict[str, Any],
        thinking_engine: SNNCore,  # System 1 (BitSpikeMamba/SFormer)
        reasoning_engine: ReasoningEngine,  # System 2 (CoT/RAG)
        meta_cognitive_snn: MetaCognitiveSNN,
        astrocyte_network: AstrocyteNetwork,
        global_workspace: GlobalWorkspace,
        actuator: Actuator,
        sleep_manager: Optional[SleepConsolidator] = None,
        device: str = "cpu"
    ):
        logger.info("🧠 Initializing Artificial Brain Kernel v21.0 [Advanced AGI Mode]...")
        self.config = config
        self.device = device
        self.event_bus = AsyncEventBus()
        
        # --- Cognitive Hierarchies ---
        self.system1 = thinking_engine
        self.system2 = reasoning_engine
        self.meta_cognition = meta_cognitive_snn
        self.astrocyte = astrocyte_network
        self.workspace = global_workspace
        self.actuator = actuator
        self.sleep_manager = sleep_manager

        # --- Runtime State ---
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.state = "AWAKE"
        self.thought_buffer: List[str] = []

    async def start(self) -> None:
        """全コグニティブプロセスの非同期開始"""
        self.running = True
        self.tasks = [
            asyncio.create_task(self._perception_to_thought_worker()),
            asyncio.create_task(self._meta_cognition_worker()),
            asyncio.create_task(self._homeostasis_worker()),
            asyncio.create_task(self._execution_worker())
        ]
        logger.info("🚀 Brain Kernel is now fully operational.")
        await asyncio.gather(*self.tasks)

    async def _perception_to_thought_worker(self) -> None:
        """知覚入力を受け取り、System 1 で即座に処理する（直感的推論）"""
        input_queue = self.event_bus.subscribe("SENSORY_INPUT")
        while self.running:
            _, raw_input = await input_queue.get()
            
            # Astrocyteによるエネルギーチェック（発火率が高い場合は抑制）
            if self.astrocyte.energy < self.config.get("min_energy_threshold", 50.0):
                logger.warning("⚠️ Low energy! System 1 processing throttled.")
                await asyncio.sleep(0.1)
                continue

            # System 1: 高速SNN推論 (BitSpike/SFormer)
            with torch.no_grad():
                # 入力形式をモデルに合わせて変換
                s1_output = self.system1(raw_input)
                
            # メタ認知へ「驚き」の判定を依頼
            await self.event_bus.publish("RAW_THOUGHT", {"source": "S1", "data": s1_output})

    async def _meta_cognition_worker(self) -> None:
        """思考を監視し、必要に応じて System 2 (熟慮) を起動する"""
        thought_queue = self.event_bus.subscribe("RAW_THOUGHT")
        while self.running:
            _, thought_data = await thought_queue.get()
            
            # 不確実性(Entropy)またはSurpriseの計算
            uncertainty = self.meta_cognition.estimate_uncertainty(thought_data["data"])
            
            if uncertainty > self.config.get("reasoning_trigger", 0.7):
                logger.info(f"🤔 High uncertainty ({uncertainty:.2f}). Activating System 2 Reasoning...")
                
                # System 2: 熟慮 (RAG + Chain-of-Thought)
                # 重い処理のためExecutorで実行
                loop = asyncio.get_running_loop()
                s2_result = await loop.run_in_executor(
                    None, self.system2.reason, thought_data["data"]
                )
                
                await self.event_bus.publish("FINAL_DECISION", s2_result, priority=5)
            else:
                # System 1 の結果をそのまま採用
                await self.event_bus.publish("FINAL_DECISION", thought_data["data"], priority=10)

    async def _homeostasis_worker(self) -> None:
        """Astrocyte Network による代謝管理とエネルギーバランスの調整"""
        while self.running:
            # 現在の発火率を取得
            firing_rates = self.system1.get_firing_rates()
            
            # 恒常性維持: 発火率が高すぎる場合はシナプス伝達効率を一時的に下げる
            for layer_name, rate in firing_rates.items():
                if rate > self.config.get("max_firing_rate", 0.5):
                    self.astrocyte.modulate_gain(layer_name, 0.8) # 抑制
                else:
                    self.astrocyte.modulate_gain(layer_name, 1.0) # 復帰

            self.astrocyte.step()
            
            # 疲労物質が溜まったら強制睡眠
            if self.astrocyte.fatigue_toxin > self.config.get("sleep_threshold", 90.0):
                await self.perform_sleep_cycle()
            
            await asyncio.sleep(1.0)

    async def _execution_worker(self) -> None:
        """グローバルワークスペースで承認された意思決定をアクチュエータへ送る"""
        decision_queue = self.event_bus.subscribe("FINAL_DECISION")
        while self.running:
            _, decision = await decision_queue.get()
            
            # Global Workspace による放送（意識への上り）
            self.workspace.broadcast(decision)
            
            # アクチュエータへの指令実行
            self.actuator.execute(decision)

    async def perform_sleep_cycle(self) -> None:
        """睡眠サイクル: 記憶の固定化とエネルギーの完全回復"""
        logger.info("💤 Fatigue limit reached. Entering sleep cycle...")
        self.state = "SLEEPING"
        
        if self.sleep_manager:
            # 昼間の思考トレースをSNN重みに蒸留
            await asyncio.to_thread(self.sleep_manager.consolidate_memory)
        
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(100.0)
        self.system1.reset_state()
        
        self.state = "AWAKE"
        logger.info("☀️ Morning has come. Brain refreshed.")

    def get_status(self) -> Dict[str, Any]:
        """ヘルスチェック v3.1 準拠のステータスレポート"""
        return {
            "status": "HEALTHY" if self.astrocyte.fatigue_toxin < 50 else "DEGRADED",
            "state": self.state,
            "energy_percent": (self.astrocyte.energy / 1000.0) * 100.0,
            "active_tasks": len([t for t in self.tasks if not t.done()]),
            "system_load": self.astrocyte.get_load_metrics()
        }

    def stop(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()
        logger.info("🛑 Brain Kernel shut down.")
