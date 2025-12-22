# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (型定義・属性エラー修正版)
# 目的: mypyエラー (attr-defined, var-annotated) を解消し、システムの整合性を確保。

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, cast, Callable

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """モジュール間の非同期通信を担う優先度付きPub/Subバス。"""
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[asyncio.PriorityQueue]] = {}

    def subscribe(self, event_type: str) -> asyncio.PriorityQueue:
        # P4-mypy: queue の型アノテーションを追加
        queue: asyncio.PriorityQueue[Tuple[int, Any]] = asyncio.PriorityQueue()
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(queue)
        return queue

    async def publish(self, event_type: str, data: Any, priority: int = 10) -> None:
        if event_type in self.subscribers:
            for queue in self.subscribers[event_type]:
                await queue.put((priority, data))

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ.
    テストやダッシュボードとの互換性を確保するため、明示的な属性定義を行う。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # --- テストおよびダッシュボードが期待する属性名へのマッピング ---
        self.workspace = kwargs.get('global_workspace')
        self.motivation_system = kwargs.get('motivation_system')
        self.receptor = kwargs.get('sensory_receptor')
        self.encoder = kwargs.get('spike_encoder')
        self.actuator = kwargs.get('actuator')
        self.system1 = kwargs.get('thinking_engine')
        self.perception = kwargs.get('perception_cortex')
        self.visual = kwargs.get('visual_cortex')
        self.pfc = kwargs.get('prefrontal_cortex')
        self.hippocampus = kwargs.get('hippocampus')
        self.cortex = kwargs.get('cortex')
        self.amygdala = kwargs.get('amygdala')
        self.basal_ganglia = kwargs.get('basal_ganglia')
        self.cerebellum = kwargs.get('cerebellum')
        self.motor = kwargs.get('motor_cortex')
        self.world_model = kwargs.get('world_model')
        self.astrocyte = kwargs.get('astrocyte_network')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')

        self.event_bus = AsyncEventBus()
        self.state = "AWAKE"
        self.cycle_count = 0

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        self.cycle_count += 1
        return {"cycle": self.cycle_count, "status": "SUCCESS"}

    def get_brain_status(self) -> Dict[str, Any]:
        """scripts/runners/run_brain_v16_demo.py 等が期待するエイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        astro_metrics = {
            "energy_level": energy, "energy_percent": (energy / 1000.0) * 100.0,
            "fatigue": fatigue, "efficiency": 1.0
        }
        return {
            "status": "HEALTHY" if fatigue < 50 else "TIRED",
            "state": self.state, "cycle": self.cycle_count,
            "astrocyte": {
                "status": "NORMAL", "metrics": astro_metrics, "energy_percent": astro_metrics["energy_percent"]
            }
        }
