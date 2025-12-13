# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampus Module v14.2 (Content Retrieval Fix)
# Description:
#   海馬（短期・作業記憶）モジュール。
#   知覚パターンとの類似性に基づき、関連するエピソード記憶を検索し、
#   その「内容」をグローバルワークスペースへ提供する。

from typing import List, Dict, Any
from collections import deque
import torch
import logging
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class Hippocampus:
    def __init__(self, workspace: GlobalWorkspace, capacity: int = 100):
        self.workspace = workspace
        self.capacity = capacity
        # エピソードを保持するリングバッファ
        self.working_memory: deque = deque(maxlen=capacity)
        logger.info(f"🧠 Hippocampus (Working Memory) initialized with capacity {capacity}.")

    def evaluate_relevance_and_upload(self, perception_features: torch.Tensor):
        """
        現在の知覚特徴量と過去の記憶を比較し、関連性が高ければワークスペースへ記憶を提示する。
        """
        # デフォルトは関連性なし
        salience = 0.0 
        memory_data = None
        
        if self.working_memory:
            # 直近のエピソード群と比較 (本来は全検索やインデックス検索だが、簡易的に直近5件)
            recent_episodes = self.retrieve_recent_episodes(5)
            
            best_similarity = -1.0
            best_episode = None
            
            for episode in recent_episodes:
                recent_features = episode.get('features')
                
                # 特徴量が存在し、かつテンソル型である場合のみ比較
                if recent_features is not None and isinstance(recent_features, torch.Tensor):
                    try:
                        # 形状を合わせてコサイン類似度を計算
                        f1 = perception_features.flatten()
                        f2 = recent_features.flatten()
                        
                        # 次元数が一致する場合のみ計算 (簡易実装)
                        if f1.shape[0] == f2.shape[0]:
                            similarity = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_episode = episode
                    except Exception as e:
                        # 計算エラーは無視して次へ
                        pass

            # 類似度が閾値(例えば0.6)を超えた場合、その記憶を「想起」候補とする
            if best_episode and best_similarity > 0.6:
                salience = best_similarity  # 類似度が高いほど顕著性（注意の引きやすさ）を高く設定
                
                # エピソード全体ではなく、意識に必要な情報を抽出してパッキング
                memory_data = {
                    "type": "recalled_memory",
                    "content": best_episode.get('content'), # 元の入力内容（テキストや画像の説明など）
                    "thought": best_episode.get('thought'), # 当時どう考えたか
                    "relevance_score": best_similarity,
                    "timestamp_diff": "recent" # 本来は時間差を計算
                }
                
                logger.debug(f"Hippocampus recalled memory with relevance {salience:.2f}")

        if memory_data:
            self.workspace.upload_to_workspace(
                source="hippocampus", 
                data=memory_data, 
                salience=salience
            )

    def store_episode(self, episode: Dict[str, Any]):
        """エピソード（経験）を一時記憶に保存"""
        self.working_memory.append(episode)

    def retrieve_recent_episodes(self, num_episodes: int = 5) -> List[Dict[str, Any]]:
        """直近のエピソードを取得"""
        if num_episodes <= 0: return []
        num = min(num_episodes, len(self.working_memory))
        # dequeの右側が最新なので、逆順で取得
        return [self.working_memory[-(i+1)] for i in range(num)]
    
    def get_and_clear_episodes_for_consolidation(self) -> List[Dict[str, Any]]:
        """睡眠時の固定化のために全エピソードを取得してクリアする"""
        episodes = list(self.working_memory)
        self.clear_memory()
        return episodes

    def clear_memory(self):
        self.working_memory.clear()
