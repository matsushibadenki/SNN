# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel v21.6 (Robust Async Edition)

import asyncio
import logging
from typing import Dict, Any, List, Optional, cast
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """SNNベース 人工脳アーキテクチャ。機能を維持しつつ例外処理を強化。"""
    def __init__(self, **components):
        # 動的なコンポーネント登録
        for name, obj in components.items():
            setattr(self, name, obj)
        
        # 互換用エイリアス
        self.workspace = components.get('global_workspace')
        self.system1 = components.get('thinking_engine')
        self.system2 = components.get('reasoning_engine')
        
        self.event_bus = AsyncEventBus()
        self.state = "AWAKE"
        self.running = False
        self.tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        self.running = True
        worker_configs = [
            self._perception_worker,
            self._thought_worker,
            self._homeostasis_worker
        ]
        self.tasks = [asyncio.create_task(w()) for w in worker_configs]
        logger.info("Brain services started.")
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Brain services shutting down.")

    async def _perception_worker(self) -> None:
        input_queue = self.event_bus.subscribe("SENSORY_INPUT")
        while self.running:
            try:
                _, raw_data = await input_queue.get()
                # 思考エンジンによる推論
                output = self.system1(raw_data)
                await self.event_bus.publish("RAW_THOUGHT", output)
            except Exception as e:
                logger.error(f"Perception Error: {e}")

    async def _homeostasis_worker(self) -> None:
        """恒常性維持（代謝と睡眠）"""
        while self.running:
            if hasattr(self, 'astrocyte'):
                self.astrocyte.step()
                if getattr(self.astrocyte, 'fatigue_toxin', 0.0) > 90.0:
                    await self.perform_sleep_cycle()
            await asyncio.sleep(1.0)

    async def perform_sleep_cycle(self) -> None:
        """非同期睡眠プロセス。機能を完全に保持。"""
        self.state = "SLEEPING"
        logger.info("Entering sleep cycle...")
        if hasattr(self, 'sleep_manager') and self.sleep_manager:
            await asyncio.to_thread(self.sleep_manager.consolidate_memory)
        if hasattr(self, 'astrocyte'):
            self.astrocyte.replenish_energy(1000.0)
        self.state = "AWAKE"

    def stop(self) -> None:
        self.running = False
        for t in self.tasks: t.cancel()
