# ファイルパス: snn_research/cognitive_architecture/async_brain_kernel.py
# 日本語タイトル: Async Brain Kernel v2.9 - Fixed Astrocyte Typing
# 目的・内容:
#   - DummyAstrocyteをトップレベルに定義し、AstrocyteNetworkとのインターフェース互換性を確保。
#   - self.astrocyte に Any 型ヒントを使用し、継承時の型不整合エラーを回避。

import asyncio
import logging
import time
import uuid
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple 
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BrainEvent:
    event_type: str
    source: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class AsyncEventBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable[[BrainEvent], Awaitable[None]]]] = {}
        self.event_queue: asyncio.PriorityQueue[Tuple[float, float, BrainEvent]] = asyncio.PriorityQueue()
        self.history: deque[BrainEvent] = deque(maxlen=100)
        self.is_running: bool = True

    def subscribe(self, event_type: str, callback: Callable[[BrainEvent], Awaitable[None]]) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def publish(self, event: BrainEvent) -> None:
        if not self.is_running:
            return
        await self.event_queue.put((-event.priority, event.timestamp, event))
        self.history.append(event)
        logger.debug(f"📨 Pub: {event.event_type} (Pri: {event.priority})")

    async def dispatch_worker(self) -> None:
        while self.is_running:
            try:
                try:
                    priority_tuple = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                event = priority_tuple[2]
                if event.event_type in self.subscribers:
                    tasks = [callback(event) for callback in self.subscribers[event.event_type]]
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for res in results:
                            if isinstance(res, Exception):
                                logger.error(f"❌ Async Task Failed: {res}", exc_info=True)
                self.event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Dispatch Error: {e}", exc_info=True)

class CognitiveModuleWrapper:
    """非同期・同期両対応のモジュールラッパー"""
    def __init__(self, name: str, module_instance: Any, bus: AsyncEventBus, executor: ThreadPoolExecutor):
        self.name = name
        self.module = module_instance
        self.bus = bus
        self.executor = executor
        self.last_active_time = 0.0

    async def process(self, input_data: Any) -> Any:
        self.last_active_time = time.time()
        
        # 1. モジュールが非同期メソッド (async def process) を持つ場合
        if hasattr(self.module, 'process') and asyncio.iscoroutinefunction(self.module.process):
            return await self.module.process(input_data)

        # 2. 同期メソッドの場合、スレッドプールで実行
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._sync_process_wrapper, input_data)

    def _sync_process_wrapper(self, input_data: Any) -> Any:
        try:
            if hasattr(self.module, 'process'):
                return self.module.process(input_data)
            elif hasattr(self.module, 'forward'): # PyTorch Module
                with torch.no_grad():
                    return self.module.forward(input_data)
            elif hasattr(self.module, '__call__'):
                return self.module(input_data)
        except Exception as e:
            logger.error(f"💥 Module Execution Error in '{self.name}': {e}")
            raise e
        return None

# DummyAstrocyteをトップレベルに移動し、機能拡充
class DummyAstrocyte:
    def request_resource(self, name: str, amount: float) -> bool: return True
    def step(self) -> None: pass
    def get_energy_level(self) -> float: return 1.0
    def get_diagnosis_report(self) -> Dict[str, Any]: return {"metrics": {"current_energy": 1000.0, "fatigue_index": 0.0}}
    # 互換性用メソッド
    def consume_energy(self, source: str, amount: float = 5.0) -> None: pass
    def request_compute_boost(self) -> bool: return True
    def log_fatigue(self, amount: float) -> None: pass
        
class AsyncArtificialBrain:
    # 柔軟な型ヒントを使用
    astrocyte: Any 

    def __init__(
        self,
        modules: Optional[Dict[str, Any]] = None,
        astrocyte: Optional[Any] = None,
        web_crawler: Optional[Any] = None,
        distillation_manager: Optional[Any] = None,
        max_workers: int = 4
    ):
        logger.info("🚀 Booting Async Brain Kernel v2.9...")
        self.bus = AsyncEventBus()
        self.modules: Dict[str, CognitiveModuleWrapper] = {}
        
        # アストロサイト（エネルギー管理）がない場合はダミーを作成
        if astrocyte is None:
             self.astrocyte = DummyAstrocyte()
        else:
            self.astrocyte = astrocyte
            
        self.web_crawler = web_crawler
        self.distillation_manager = distillation_manager
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BrainWorker")
        
        if modules:
            for name, instance in modules.items():
                self.modules[name] = CognitiveModuleWrapper(name, instance, self.bus, self.executor)
            self.reflex_module = modules.get("reflex_module")
        else:
            self.reflex_module = None

        self.state = "BOOTING"
        self._shutdown_event = asyncio.Event()
        self.background_tasks: List[asyncio.Task] = []
        
        self._setup_wiring()

    def connect_adapter(self, adapter: Any, name: str = "adapter_interface"):
        """外部アダプターをモジュールとして接続する"""
        logger.info(f"🔗 Connecting Adapter: {name}")
        self.modules[name] = CognitiveModuleWrapper(name, adapter, self.bus, self.executor)

    async def set_mode(self, mode: str):
        """脳の状態モードを変更する"""
        logger.info(f"🔄 Mode Change: {self.state} -> {mode.upper()}")
        self.state = mode.upper()
        
        await self.bus.publish(BrainEvent(
            event_type="MODE_CHANGED",
            source="kernel",
            payload={"new_mode": self.state},
            priority=10.0
        ))

    def get_status(self) -> Dict[str, Any]:
        """現在のステータスとメトリクスを取得する"""
        report = self.astrocyte.get_diagnosis_report()
        return {
            "state": self.state,
            "mode": self.state,
            "metrics": report.get("metrics", {})
        }
    
    @property
    def astrocyte_network(self):
        """Demoスクリプトが直接アクセスする場合のプロパティ"""
        return self.astrocyte

    def _setup_wiring(self):
        self.bus.subscribe("SENSORY_INPUT", self._on_sensory_input)
        self.bus.subscribe("PERCEPTION_DONE", self._on_perception_done)
        self.bus.subscribe("EMOTIONAL_REACTION", self._on_emotional_reaction)
        self.bus.subscribe("CONSCIOUS_BROADCAST", self._on_conscious_broadcast)
        self.bus.subscribe("THOUGHT_COMPLETE", self._on_thought_complete)
        self.bus.subscribe("PLAN_GENERATED", self._on_plan_generated)
        self.bus.subscribe("KNOWLEDGE_GAP_DETECTED", self._on_knowledge_gap_detected)
        self.bus.subscribe("KNOWLEDGE_UPDATED", self._on_knowledge_updated)
        self.bus.subscribe("ACTION_COMMAND", self._on_action_command)


    async def _on_sensory_input(self, event: BrainEvent):
        payload = event.payload
        
        if self.reflex_module is not None and isinstance(payload, torch.Tensor):
            try:
                reflex_instance = self.modules["reflex_module"].module
                if hasattr(reflex_instance, "forward"):
                    with torch.no_grad():
                        action_id, confidence = reflex_instance.forward(payload)
                        
                    if action_id is not None:
                        logger.warning(f"⚡ REFLEX TRIGGERED! Action: {action_id} (Conf: {confidence:.2f})")
                        await self.bus.publish(BrainEvent(
                            event_type="ACTION_COMMAND",
                            source="reflex_module",
                            payload={"action_id": action_id, "type": "REFLEX"},
                            priority=100.0
                        ))
            except Exception as e:
                logger.warning(f"Reflex logic failed: {e}")

        tasks = []
        if "visual_cortex" in self.modules and (isinstance(payload, torch.Tensor) or "image" in str(type(payload))):
            tasks.append(self._run_module("visual_cortex", payload, "PERCEPTION_DONE"))
        
        if isinstance(payload, str):
            if "language_area" in self.modules:
                 tasks.append(self._run_module("language_area", payload, "PERCEPTION_DONE"))
            else:
                await self.bus.publish(BrainEvent(
                    event_type="PERCEPTION_DONE",
                    source="sensory_gateway",
                    payload=payload,
                    metadata={"modality": "text"},
                    priority=5.0
                ))
        
        if "agent" in self.modules:
             tasks.append(self._run_module("agent", payload, "ACTION_COMMAND"))
             
        for t in tasks:
            asyncio.create_task(t)

    async def _on_perception_done(self, event: BrainEvent):
        payload = event.payload
        if "amygdala" in self.modules:
            asyncio.create_task(self._run_module("amygdala", payload, "EMOTIONAL_REACTION"))

        cost = 1.0
        if self.astrocyte.request_resource("attention", cost):
            await self.bus.publish(BrainEvent(
                event_type="CONSCIOUS_BROADCAST",
                source="global_workspace",
                payload=payload,
                metadata=event.metadata,
                priority=event.priority + 1.0
            ))

    async def _on_emotional_reaction(self, event: BrainEvent):
        emotion_data = event.payload 
        if "system1" in self.modules and isinstance(emotion_data, dict):
            adapter = self.modules["system1"].module
            if hasattr(adapter, "update_mood"):
                adapter.update_mood(
                    valence=emotion_data.get("valence", 0.0),
                    arousal=emotion_data.get("arousal", 0.0)
                )

    async def _on_conscious_broadcast(self, event: BrainEvent):
        payload = event.payload
        meta = event.metadata
        
        trigger_system2 = meta.get("trigger_system2", False)
        needs_planning = meta.get("needs_planning", False)
        
        tasks = []

        if needs_planning and "planner" in self.modules:
            logger.info("🧠 Routing to Planner")
            tasks.append(self._run_module("planner", payload, "PLAN_GENERATED"))

        elif trigger_system2 and "reasoning_engine" in self.modules:
            logger.info("🤔 Routing to Reasoning Engine")
            tasks.append(self._run_module("reasoning_engine", payload, "THOUGHT_COMPLETE"))
            
        elif "system1" in self.modules:
            tasks.append(self._run_module("system1", payload, "ACTION_COMMAND"))
            
        for t in tasks:
            asyncio.create_task(t)

    async def _on_thought_complete(self, event: BrainEvent):
        result = event.payload
        await self.bus.publish(BrainEvent(
            event_type="ACTION_COMMAND",
            source="reasoning_engine",
            payload=result,
            priority=8.0
        ))
    
    async def _on_plan_generated(self, event: BrainEvent):
        plan = event.payload
        await self.bus.publish(BrainEvent(
            event_type="ACTION_COMMAND",
            source="planner",
            payload=plan,
            priority=8.0
        ))

    async def _on_knowledge_gap_detected(self, event: BrainEvent):
        if not self.web_crawler or not self.distillation_manager:
            logger.warning("⚠️ Web Learning modules missing.")
            return

        payload = event.payload
        query_topic: str = str(payload.get("topic", str(payload))) if isinstance(payload, dict) else str(payload)

        logger.info(f"🔍 Knowledge Gap: '{query_topic}'. Initiating Web Learning...")
        prev_state = self.state
        self.state = "LEARNING"
        asyncio.create_task(self._execute_web_learning(query_topic, prev_state))

    async def _execute_web_learning(self, topic: str, prev_state: str):
        if self.web_crawler is None or self.distillation_manager is None:
            self.state = prev_state
            return

        try:
            logger.info(f"🌐 Crawling web for: {topic}")
            start_url = f"https://www.google.com/search?q={topic}" 
            crawler = self.web_crawler
            
            if hasattr(crawler, 'crawl') and asyncio.iscoroutinefunction(crawler.crawl):
                crawled_path = await crawler.crawl(start_url=start_url, max_pages=3)
            else:
                loop = asyncio.get_running_loop()
                crawled_path = await loop.run_in_executor(
                    self.executor, 
                    lambda: crawler.crawl(start_url=start_url, max_pages=3)
                )

            if not crawled_path:
                logger.warning("❌ Crawling failed.")
                return

            logger.info("🧠 Distilling knowledge...")
            await self.distillation_manager.run_on_demand_pipeline(
                task_description=topic,
                unlabeled_data_path=crawled_path,
                force_retrain=True
            )
            
            await self.bus.publish(BrainEvent(
                event_type="KNOWLEDGE_UPDATED",
                source="hippocampus_web_loader",
                payload={"topic": topic, "status": "success"},
                priority=10.0
            ))

        except Exception as e:
            logger.error(f"💥 Active Web Learning Failed: {e}", exc_info=True)
        finally:
            self.state = prev_state

    async def _on_knowledge_updated(self, event: BrainEvent):
        logger.info("🎉 Knowledge Updated.")

    async def _on_action_command(self, event: BrainEvent):
        if "actuator" in self.modules:
            await self.modules["actuator"].process(event.payload)
            if "REFLEX" in str(event.payload):
                self.astrocyte.request_resource("stress_response", 20.0)

    async def _run_module(self, module_name: str, input_data: Any, output_event_type: Optional[str]):
        try:
            if module_name not in self.modules:
                return

            wrapper = self.modules[module_name]
            
            energy_cost = 5.0
            if module_name in ["reasoning_engine", "planner"]:
                energy_cost = 20.0
                
            if not self.astrocyte.request_resource(module_name, energy_cost):
                logger.warning(f"🔋 Low Energy: Skipping {module_name}")
                return
            
            result = await wrapper.process(input_data)
            
            if module_name == "reflex_module" and isinstance(result, tuple):
                 action_id, confidence = result
                 if action_id is None:
                     result = None 
                 else:
                     result = {"action_id": action_id, "type": "REFLEX"}

            if result is not None and output_event_type:
                meta = {}
                if isinstance(result, dict) and "metadata" in result:
                    meta = result.pop("metadata")

                await self.bus.publish(BrainEvent(
                    event_type=output_event_type,
                    source=module_name,
                    payload=result,
                    metadata=meta,
                    priority=1.0
                ))
        except Exception as e:
            logger.error(f"💥 Error in module {module_name}: {e}")

    async def start(self):
        self.state = "RUNNING"
        self.bus.is_running = True
        logger.info("🧠 Async Brain Kernel Started.")
        self.background_tasks.append(asyncio.create_task(self.bus.dispatch_worker()))
        self.background_tasks.append(asyncio.create_task(self._homeostasis_loop()))

    async def stop(self):
        logger.info("💤 Stopping Brain Kernel...")
        self.state = "SHUTDOWN"
        self._shutdown_event.set()
        self.bus.is_running = False
        for task in self.background_tasks:
            task.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def _homeostasis_loop(self):
        while not self._shutdown_event.is_set():
            try:
                self.astrocyte.step()
                if self.astrocyte.get_energy_level() < 0.1 and self.state != "SLEEP":
                    logger.warning("⚠️ Critical Energy! SLEEP mode activated.")
                    self.state = "SLEEP"
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Homeostasis Error: {e}")

    async def receive_input(self, data: Any):
        await self.bus.publish(BrainEvent(
            event_type="SENSORY_INPUT",
            source="external_sensor",
            payload=data,
            priority=10.0
        ))

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        active = [n for n, w in self.modules.items() if (now - w.last_active_time) < 5.0]
        recent = [
            {
                "time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
                "type": e.event_type,
                "payload": str(e.payload)[:30]
            }
            for e in list(self.bus.history)[-10:]
        ]
        return {
            "state": self.state,
            "active_modules": active,
            "recent_events": recent,
            "energy_levels": self.astrocyte.get_diagnosis_report()["metrics"]
        }