# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# (修正: NoneTypeエラー回避)

from typing import List, Dict, Any
from collections import deque
import torch
from .global_workspace import GlobalWorkspace

class Hippocampus:
    def __init__(self, workspace: GlobalWorkspace, capacity: int = 100):
        self.workspace = workspace
        self.capacity = capacity
        self.working_memory: deque = deque(maxlen=capacity)
        print(f"🧠 海馬（ワーキングメモリ）モジュールが初期化されました。")

    def evaluate_relevance_and_upload(self, perception_features: torch.Tensor):
        salience = 0.8 
        relevance_info = {"type": "memory_relevance", "relevance": 0.0, "details": "No existing memories."}

        if self.working_memory:
            recent_episodes = self.retrieve_recent_episodes(1)
            if recent_episodes:
                recent_episode = recent_episodes[0]
                
                # --- ▼ 修正: 安全なデータ取得 ▼ ---
                content = recent_episode.get('content')
                recent_features = None
                if content is not None and isinstance(content, dict):
                    recent_features = content.get('features')
                # --- ▲ 修正 ▲ ---
                
                if recent_features is not None and isinstance(recent_features, torch.Tensor):
                    try:
                        f1 = perception_features.flatten()
                        f2 = recent_features.flatten()
                        if f1.shape == f2.shape:
                            similarity = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                            salience = 1.0 - similarity
                            relevance_info = {"type": "memory_relevance", "relevance": similarity}
                    except Exception:
                         pass

        self.workspace.upload_to_workspace(source="hippocampus", data=relevance_info, salience=salience)

    def store_episode(self, episode: Dict[str, Any]):
        self.working_memory.append(episode)

    def retrieve_recent_episodes(self, num_episodes: int = 5) -> List[Dict[str, Any]]:
        if num_episodes <= 0: return []
        num = min(num_episodes, len(self.working_memory))
        return [self.working_memory[-(i+1)] for i in range(num)]
    
    def get_and_clear_episodes_for_consolidation(self) -> List[Dict[str, Any]]:
        episodes = list(self.working_memory)
        self.clear_memory()
        return episodes

    def clear_memory(self):
        self.working_memory.clear()