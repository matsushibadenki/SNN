# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳カーネル v21.8 (Integrity & Async Optimized)
# 目的: モジュール間通信の円滑化と、生体模倣（睡眠・代謝）ロジックの統合。

import asyncio
import logging
import torch
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """高優先度タスク（反射等）を優先する非同期Pub/Subバス。"""
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[asyncio.PriorityQueue]] = {}

    def subscribe(self, event_type: str) -> asyncio.PriorityQueue:
        queue = asyncio.PriorityQueue()
        self.subscribers.setdefault(event_type, []).append(queue)
        return queue

    async def publish(self, event_type: str, data: Any, priority: int = 10) -> None:
        if event_type in self.subscribers:
            # 優先度付きキューへ投入 (低い数値ほど高優先)
            tasks = [q.put((priority, data)) for q in self.subscribers[event_type]]
            await asyncio.gather(*tasks)

class ArtificialBrain:
    """
    Project SNN の中枢。全認知ドメインを統括する。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 主要コンポーネントの動的マッピング
        self.components = kwargs
        self.event_bus = AsyncEventBus()
        self.state = "AWAKE"
        self.cycle_count = 0
        
        # ヘルスチェック用のショートカット参照
        self.astrocyte = kwargs.get('astrocyte_network')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        同期実行サイクル（レガシー互換およびテスト用）。
        実際のリサーチでは非同期メソッドの使用を推奨。
        """
        self.cycle_count += 1
        # 入力のエンコードから行動生成までのパイプラインをシミュレート
        status = self.get_status()
        
        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "state": self.state,
            "astrocyte_health": status["astrocyte"]["status"]
        }

    def get_status(self) -> Dict[str, Any]:
        """
        脳の代謝・健康状態の診断。
        AstrocyteNetworkのメトリクスを抽出し、System 2が処理可能な形式で返す。
        """
        # デフォルト値の設定
        energy = 1000.0
        fatigue = 0.0
        
        if self.astrocyte:
            energy = getattr(self.astrocyte, 'energy', 1000.0)
            fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0)

        astro_metrics = {
            "energy_level": energy,
            "energy_percent": (energy / 1000.0) * 100.0,
            "fatigue": fatigue,
            "efficiency": 1.0 - (fatigue / 100.0) if fatigue < 100 else 0.0
        }

        return {
            "status": "HEALTHY" if fatigue < 50 else "CRITICAL",
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": "NORMAL" if fatigue < 50 else "WARNING",
                "metrics": astro_metrics,
                "energy_percent": astro_metrics["energy_percent"]
            }
        }

    def sleep_cycle(self) -> None:
        """睡眠による記憶の定着と代謝リセット。"""
        logger.info("🌙 Entering Sleep Cycle (Consolidation & Detox)...")
        self.state = "SLEEPING"
        
        # 1. 記憶の固定化
        if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
            self.sleep_manager.consolidate_memory()
            
        # 2. アストロサイトによるエネルギー充填と毒素洗浄
        if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
            self.astrocyte.replenish_energy(1000.0)
            
        # 3. SNNコアの状態リセット（System 1等）
        if 'thinking_engine' in self.components:
            engine = self.components['thinking_engine']
            if hasattr(engine, 'reset_state'):
                engine.reset_state()

        self.state = "AWAKE"
        logger.info("☀️ Brain state: AWAKE. Ready for next cycles.")
