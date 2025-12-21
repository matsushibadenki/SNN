# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (mypy完全対応版)
# 目的: クラス属性を明示的に宣言し、外部スクリプト（dashboard.py等）との型整合性を確保する。

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# 非同期イベントバスの定義（省略せずに維持）
class AsyncEventBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[asyncio.PriorityQueue]] = {}
    
    def subscribe(self, event_type: str) -> asyncio.PriorityQueue:
        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.subscribers.setdefault(event_type, []).append(queue)
        return queue

    async def publish(self, event_type: str, data: Any, priority: int = 10) -> None:
        if event_type in self.subscribers:
            for queue in self.subscribers[event_type]:
                await queue.put((priority, data))

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ。
    mypyが属性を追跡できるように、すべてのコンポーネントを明示的に宣言。
    """
    # 外部スクリプトがアクセスする属性を明示的に宣言
    workspace: Any
    sleep_manager: Any
    cortex: Any
    hippocampus: Any
    basal_ganglia: Any
    amygdala: Any
    pfc: Any
    motor: Any
    system1: Any
    system2: Optional[Any]
    meta_cognition: Optional[Any]
    astrocyte: Any
    event_bus: AsyncEventBus
    running: bool
    state: str
    cycle_count: int

    def __init__(self, **components: Any):
        # コンポーネントの割り当てとフォールバック
        self.workspace = components.get('global_workspace')
        self.sleep_manager = components.get('sleep_manager') or components.get('sleep_consolidator')
        self.cortex = components.get('cortex')
        self.hippocampus = components.get('hippocampus')
        self.basal_ganglia = components.get('basal_ganglia')
        self.amygdala = components.get('amygdala')
        self.pfc = components.get('prefrontal_cortex')
        self.motor = components.get('motor_cortex')
        self.system1 = components.get('thinking_engine')
        self.system2 = components.get('reasoning_engine')
        self.meta_cognition = components.get('meta_cognitive_snn')
        self.astrocyte = components.get('astrocyte_network')

        # ランタイム状態
        self.event_bus = AsyncEventBus()
        self.running = False
        self.state = "AWAKE"
        self.cycle_count = 0
        self.tasks: List[asyncio.Task] = []

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """同期API: ダッシュボード等で使用"""
        self.cycle_count += 1
        return {"status": "SUCCESS", "cycle": self.cycle_count}

    def get_brain_status(self) -> Dict[str, Any]:
        """同期API: 健康状態取得用"""
        return {"state": self.state, "cycle": self.cycle_count}

    def sleep_cycle(self) -> None:
        """同期API: sleep_cycle_demo.py 用の互換メソッド"""
        logger.info("Initiating Synchronous Sleep Cycle...")
        self.state = "SLEEPING"
        if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
            self.sleep_manager.consolidate_memory()
        self.state = "AWAKE"
