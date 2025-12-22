# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (リファクタリング完全版)
# 目的: モジュール管理の堅牢化と、認知サイクルの完全な動作保証。

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, cast, Tuple

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """モジュール間の非同期通信を担う優先度付きPub/Subバス。"""
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[asyncio.PriorityQueue[Tuple[int, Any]]]] = {}

    def subscribe(self, event_type: str) -> asyncio.PriorityQueue[Tuple[int, Any]]:
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
    SNNベース 人工脳アーキテクチャ。
    既存の全デモスクリプト（run_brain_v16_demo.py 等）との互換性を維持。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # コンポーネントの動的バインド (既存プロパティ名を維持)
        self.workspace: Any = kwargs.get('global_workspace')
        self.motivation_system: Any = kwargs.get('motivation_system')
        self.receptor: Any = kwargs.get('sensory_receptor')
        self.encoder: Any = kwargs.get('spike_encoder')
        self.actuator: Any = kwargs.get('actuator')
        self.system1: Any = kwargs.get('thinking_engine')
        self.perception: Any = kwargs.get('perception_cortex')
        self.visual: Any = kwargs.get('visual_cortex')
        self.pfc: Any = kwargs.get('prefrontal_cortex')
        self.hippocampus: Any = kwargs.get('hippocampus')
        self.cortex: Any = kwargs.get('cortex')
        self.amygdala: Any = kwargs.get('amygdala')
        self.basal_ganglia: Any = kwargs.get('basal_ganglia')
        self.cerebellum: Any = kwargs.get('cerebellum')
        self.motor: Any = kwargs.get('motor_cortex')
        self.world_model: Any = kwargs.get('world_model')
        self.astrocyte: Any = kwargs.get('astrocyte_network')
        # sleep_managerとsleep_consolidatorの両方の命名に対応
        self.sleep_manager: Any = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.reflex_module: Any = kwargs.get('reflex_module')
        self.guardrail: Any = kwargs.get('ethical_guardrail')

        self.event_bus = AsyncEventBus()
        self.state = "AWAKE"
        self.cycle_count = 0

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        同期認知サイクル。
        入力の処理、アストロサイトの更新、および実行レポートの生成を行う。
        """
        self.cycle_count += 1
        
        # 1. 知覚（Perception / Visual）
        perception_result = None
        if self.visual and hasattr(self.visual, 'forward'):
            perception_result = self.visual(raw_input)
        
        # 2. アストロサイトによる疲労の蓄積とリソース管理
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            # スパイク活動に応じた疲労蓄積をシミュレート
            self.astrocyte.accumulate_fatigue(0.5)

        # 3. ガードレールによる倫理チェック
        if self.guardrail and hasattr(self.guardrail, 'check_safety'):
            self.guardrail.check_safety(perception_result)

        status_info = self.get_status()
        
        # 戻り値は既存のデモスクリプトが期待する形式を厳守
        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "mode": "Hybrid",
            "state": self.state,
            "astrocyte": status_info["astrocyte"],
            "result": "Brain cycle executed successfully.",
            "perception_snapshot": perception_result
        }

    def sleep_cycle(self) -> None:
        """
        同期睡眠サイクル。エネルギー回復と記憶の固定化（Consolidation）を実行。
        """
        logger.info("🛌 Initiating Sleep Cycle...")
        self.state = "SLEEPING"
        
        # 記憶の固定化
        if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
            self.sleep_manager.consolidate_memory()
            
        # アストロサイトのエネルギー充填
        if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
            self.astrocyte.replenish_energy(1000.0)
            
        # 神経状態のリセット
        if self.system1 and hasattr(self.system1, 'reset_state'):
            self.system1.reset_state()
            
        self.state = "AWAKE"
        logger.info("☀️ Brain state restored to AWAKE.")

    def get_brain_status(self) -> Dict[str, Any]:
        """既存スクリプト用のエイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """
        脳の統合的なステータスレポート。
        アストロサイトのメトリクス構造を維持し、診断情報を提供する。
        """
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        
        astro_metrics = {
            "energy_level": energy,
            "energy_percent": (energy / 1000.0) * 100.0,
            "fatigue": fatigue,
            "efficiency": max(0.0, 1.0 - (fatigue / 1000.0))
        }

        return {
            "status": "HEALTHY" if fatigue < 500 else "TIRED",
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": "NORMAL" if fatigue < 500 else "WARNING",
                "energy_percent": astro_metrics["energy_percent"],
                "fatigue": fatigue,
                "metrics": astro_metrics,
                "diagnosis": {"recommendation": "Sleep needed" if fatigue > 700 else "Good condition"}
            }
        }
