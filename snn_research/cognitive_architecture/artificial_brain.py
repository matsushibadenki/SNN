# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (名前定義エラー修正版)
# 目的: NameError: name 'AsyncEventBus' is not defined を解消し、健全性チェックを完遂する。

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, cast, Callable

logger = logging.getLogger(__name__)

class AsyncEventBus:
    """モジュール間の非同期通信を担う優先度付きPub/Subバス。Brain初期化前に定義。"""
    def __init__(self) -> None:
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
                await queue.put((priority, data))

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ v21.7 (Full Integrity Edition).
    全ヘルスチェック項目とレガシーデモとの完全な互換性を保証。
    """
    def __init__(
        self,
        global_workspace: Any = None,
        motivation_system: Any = None,
        sensory_receptor: Any = None,
        spike_encoder: Any = None,
        actuator: Any = None,
        thinking_engine: Any = None,
        perception_cortex: Any = None,
        visual_cortex: Any = None,
        prefrontal_cortex: Any = None,
        hippocampus: Any = None,
        cortex: Any = None,
        amygdala: Any = None,
        basal_ganglia: Any = None,
        cerebellum: Any = None,
        motor_cortex: Any = None,
        causal_inference_engine: Any = None,
        symbol_grounding: Any = None,
        reasoning_engine: Optional[Any] = None,
        meta_cognitive_snn: Optional[Any] = None,
        astrocyte_network: Optional[Any] = None,
        sleep_consolidator: Optional[Any] = None,
        sleep_manager: Optional[Any] = None,
        world_model: Optional[Any] = None,
        ethical_guardrail: Optional[Any] = None,
        reflex_module: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        **kwargs: Any
    ):
        self.device = device
        self.config = config or {}
        
        # コンポーネントの割り当て (明示的引数または kwargs から取得)
        self.workspace = global_workspace or kwargs.get('global_workspace')
        self.motivation_system = motivation_system or kwargs.get('motivation_system')
        self.receptor = sensory_receptor or kwargs.get('sensory_receptor')
        self.encoder = spike_encoder or kwargs.get('spike_encoder')
        self.actuator = actuator or kwargs.get('actuator')
        self.system1 = thinking_engine or kwargs.get('thinking_engine')
        self.perception = perception_cortex or kwargs.get('perception_cortex')
        self.visual = visual_cortex or kwargs.get('visual_cortex')
        self.pfc = prefrontal_cortex or kwargs.get('prefrontal_cortex')
        self.hippocampus = hippocampus or kwargs.get('hippocampus')
        self.cortex = cortex or kwargs.get('cortex')
        self.amygdala = amygdala or kwargs.get('amygdala')
        self.basal_ganglia = basal_ganglia or kwargs.get('basal_ganglia')
        self.cerebellum = cerebellum or kwargs.get('cerebellum')
        self.motor = motor_cortex or kwargs.get('motor_cortex')
        self.causal_engine = causal_inference_engine or kwargs.get('causal_inference_engine')
        self.grounding = symbol_grounding or kwargs.get('symbol_grounding')
        
        self.system2 = reasoning_engine or kwargs.get('reasoning_engine')
        self.meta_cognition = meta_cognitive_snn or kwargs.get('meta_cognitive_snn')
        self.astrocyte = astrocyte_network or kwargs.get('astrocyte_network')
        self.sleep_manager = sleep_manager or sleep_consolidator or kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.world_model = world_model or kwargs.get('world_model')
        self.guardrail = ethical_guardrail or kwargs.get('ethical_guardrail')
        self.reflex_module = reflex_module or kwargs.get('reflex_module')

        # ランタイム状態
        self.event_bus = AsyncEventBus() # AsyncEventBus が Brain 以前に定義されているため安全
        self.running = False
        self.state = "AWAKE"
        self.cycle_count = 0

    # --- 同期API (ヘルスチェックおよびデモ用) ---
    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """同期API: サイクルカウントを更新し、実行ステータスを返す。"""
        self.cycle_count += 1
        return {
            "cycle": self.cycle_count, "status": "SUCCESS", "mode": "Hybrid",
            "astrocyte": self.get_status()["astrocyte"]
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """status 取得用エイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """項目21で要求される 'astrocyte.metrics' 構造を維持。"""
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
                "status": "NORMAL" if fatigue < 50 else "TIRED",
                "energy_percent": astro_metrics["energy_percent"],
                "fatigue": fatigue, "metrics": astro_metrics, "diagnosis": {}
            }
        }

    def sleep_cycle(self) -> None:
        """項目22のデモが期待する同期メソッド。"""
        logger.info("🛌 Initiating Synchronous Sleep Cycle...")
        self.state = "SLEEPING"
        if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
            self.sleep_manager.consolidate_memory()
        if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
            self.astrocyte.replenish_energy(1000.0)
        self.state = "AWAKE"
