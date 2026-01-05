# ファイルパス: snn_research/agent/autonomous_agent.py
# Title: Autonomous Agent
# 修正: name, handle_task追加, WebCrawler/UniversalEncoder再定義修正, context_buffer型追加, MetaCognitiveAgent定義

import torch
import yaml
import os
import asyncio
from typing import Dict, Any, List, Optional

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent

# mypy redefinition error回避のため、ImportError時のフォールバックを修正
try:
    from app.services.web_crawler import WebCrawler
except ImportError:
    class WebCrawler: # type: ignore
        def search(self, query): return [f"Result for {query}"]
        def crawl(self, start_url, max_pages=5): return "dummy_path.txt"

try:
    from snn_research.io.universal_encoder import UniversalEncoder
except ImportError:
    class UniversalEncoder: # type: ignore
        def __init__(self, d_model=64, device='cpu'): pass
        def encode_text(self, text): return torch.randn(1, 64)

class AutonomousAgent:
    """
    自律学習エージェント
    """
    def __init__(self, input_size: int, output_size: int, device: str, config_path: Optional[str] = None, name: str = "AutonomousAgent"):
        self.device = device
        self.name = name
        
        # モデル設定の読み込み
        model_config = None
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    model_config = full_config.get('model', {})
                    print(f"  ⚙️ [Autonomy] Loaded scaled config from {config_path}")
            except Exception as e:
                print(f"  ⚠️ [Autonomy] Config load failed: {e}. Using default.")
        
        if model_config is None:
            model_config = {
                'architecture': 'dsa_transformer',
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2
            }

        self.brain = ReinforcementLearnerAgent(
            input_size=input_size, 
            output_size=output_size, 
            device=device,
            model_config=model_config
        )
        
        self.crawler = WebCrawler()
        self.encoder = UniversalEncoder(d_model=model_config.get('d_model', 64), device=device)
        
        self.curiosity_threshold = 0.6
        self.boredom_counter = 0
        self.last_activity_step = 0
        self.context_buffer: List[Any] = []

    def perceive_and_act(self, state: torch.Tensor, step: int) -> int:
        self.last_activity_step = step
        action = self.brain.get_action(state)
        
        with torch.no_grad():
            confidence = 0.8 
            if torch.rand(1).item() < 0.05:
                confidence = 0.3 
        
        if confidence < self.curiosity_threshold:
            print(f"  [Autonomy] Unknown state detected (Conf: {confidence:.2f}). Triggering curiosity...")
            self._satisfy_curiosity(state)
            
        return action

    def _satisfy_curiosity(self, state_context: torch.Tensor):
        query = "What is this pattern?" 
        results = self.crawler.search(query)
        if not results:
            return

        print(f"  [Autonomy] Crawled {len(results)} new pieces of info.")
        
        trajectories = []
        for info in results[:2]: 
            encoded_info = self.encoder.encode_text(info)
            state_input = encoded_info.mean(dim=0)
            
            trajectory = {
                'spikes_history': [{'state': state_input.unsqueeze(0), 'action': 0}],
                'total_reward': 1.0
            }
            trajectories.append(trajectory)
            
        if trajectories:
            self.brain.learn_with_grpo(trajectories)
            
    def idle_routine(self):
        self.boredom_counter += 1
        if self.boredom_counter > 100:
            print("  [Autonomy] I am bored. Searching for new trends...")
            self._satisfy_curiosity(torch.tensor([])) # Dummy state
            self.boredom_counter = 0

    def save(self, path: str):
        torch.save(self.brain.model.state_dict(), path)
        
    def load(self, path: str):
        self.brain.model.load_state_dict(torch.load(path, map_location=self.device))

    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """
        EmergentSystemからのタスク委譲を処理する。
        """
        print(f"  🤖 [{self.name}] Handling task: {task_description}")
        await asyncio.sleep(0.1) # モック処理
        
        # 簡易的な成功判定
        if "fail" in task_description.lower():
            return None
        
        return {
            "status": "success",
            "model_id": "self_optimized_model_v1",
            "metrics": {"accuracy": 0.95}
        }

# エイリアス定義 (Importエラー回避)
MetaCognitiveAgent = AutonomousAgent