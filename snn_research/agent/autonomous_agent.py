# ファイルパス: snn_research/agent/autonomous_agent.py
# Title: Autonomous Agent Base - 実装版

from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
import asyncio
import torch
import json

from snn_research.distillation.model_registry import ModelRegistry
from app.services.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder
from snn_research.tools.calculator import Calculator

if TYPE_CHECKING:
    from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

class AutonomousAgent:
    calculator: Calculator

    def __init__(
        self,
        name: str,
        planner: "HierarchicalPlanner",
        model_registry: ModelRegistry,
        memory: AgentMemory,
        web_crawler: WebCrawler,
        accuracy_threshold: float = 0.6,
        energy_budget: float = 10000.0,
        **kwargs: Any
    ):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler
        self.calculator = Calculator()
        self.current_state: Dict[str, Union[str, Dict[str, Any], None]] = {
            "agent_name": name, "last_action": None, "last_result": None
        }
        self.accuracy_threshold = accuracy_threshold
        self.energy_budget = energy_budget
        self.spike_communicator = SpikeEncoderDecoder()

    def receive_and_process_spike_message(self, spike_pattern: torch.Tensor, source_agent: str):
        """スパイクメッセージを受信し、デコードしてメモリに記録する。"""
        decoded_data = self.spike_communicator.decode_data(spike_pattern)
        print(f"🤖 Agent '{self.name}' received spike message from '{source_agent}': {decoded_data}")
        
        self.memory.record_experience(
            state=self.current_state,
            action="receive_message",
            result=decoded_data,
            reward={"external": 0.1}, 
            expert_used=[],
            decision_context={"source": source_agent, "type": "communication"}
        )
    
    def execute(self, task_description: str) -> str:
        """タスクを実行する（簡易版）。"""
        print(f"🤖 Agent '{self.name}' executing task: {task_description}")
        
        expert = asyncio.run(self.find_expert(task_description))
        
        if expert:
            return f"Executed '{task_description}' using expert '{expert.get('model_id')}'."
        else:
            return f"Failed to execute '{task_description}'. No expert found."

    async def find_expert(self, task_description: str) -> Optional[Dict[str, Any]]:
        """タスクに最適な専門家モデルを検索する。"""
        safe_task_description = task_description.lower().replace(" ", "_").replace("/", "_")
        candidate_experts = await self.model_registry.find_models_for_task(safe_task_description, top_k=5)
        
        if not candidate_experts:
            return None
            
        def get_accuracy(expert: Dict[str, Any]) -> float:
            metrics = expert.get("metrics")
            if metrics and isinstance(metrics, dict):
                return float(metrics.get("accuracy", 0.0))
            return 0.0

        suitable_experts = [e for e in candidate_experts if get_accuracy(e) >= self.accuracy_threshold]
        
        if suitable_experts:
            return max(suitable_experts, key=get_accuracy)
        else:
            return candidate_experts[0]

    def learn_from_web(self, topic: str) -> str:
        """Webクローラーを使用して学習する。"""
        print(f"🤖 Agent '{self.name}' learning about '{topic}' from web...")
        start_url = f"https://www.google.com/search?q={topic}" 
        data_path = self.web_crawler.crawl(start_url=start_url, max_pages=3)
        
        if data_path:
            return f"Learned from web. Data saved to {data_path}."
        return "Failed to learn from web."

    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """タスク処理のメインルーチン。モデル検索または学習を行う。"""
        if not force_retrain:
            expert = await self.find_expert(task_description)
            if expert:
                return expert
        
        if unlabeled_data_path:
            return {"model_id": f"new_expert_{task_description}", "metrics": {"accuracy": 0.8}, "path": "dummy/path"}
            
        return None

    async def run_inference(self, model_info: Dict[str, Any], prompt: str) -> None:
        """選択されたモデルで推論を実行するシミュレーション。"""
        model_path = model_info.get("model_path") or model_info.get("path")
        print(f"🤖 Running inference on '{model_path}' with prompt: '{prompt}'")
        print(f"   -> Inference result: [Simulated Output for '{prompt}']")