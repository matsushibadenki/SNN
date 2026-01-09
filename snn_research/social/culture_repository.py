# ファイルパス: snn_research/social/culture_repository.py
# Title: Culture Repository (Memetic Store)
# Description:
# - エージェント間で共有される知識(Concept/Meme)の永続化層。
# - 成功した行動パターンや概念ベクトルを保存し、新世代のエージェントに継承させる。

import torch
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class CultureRepository:
    """
    文化リポジトリ。
    集団が得た知識を「ミーム(Meme)」として保存し、検索可能にする。
    ファイルシステム上にJSONデータベースとして永続化する。
    """
    def __init__(self, storage_path: str = "workspace/culture_db.json"):
        self.storage_path = storage_path
        self.memes: Dict[str, Any] = {} # {meme_id: {vector, description, utility, ...}}
        self.history: List[str] = []
        
        self._load()
        logger.info("📚 Culture Repository initialized.")

    def contribute_meme(self, concept_name: str, vector: torch.Tensor, description: str, utility_score: float):
        """
        新しい概念（ミーム）を文化に登録する。
        """
        # ユニークID生成
        meme_id = f"{concept_name}_{int(time.time())}"
        
        # ベクトルをリスト化してJSONシリアライズ可能にする
        vector_data = vector.cpu().tolist() if isinstance(vector, torch.Tensor) else vector
        
        entry = {
            "id": meme_id,
            "name": concept_name,
            "vector": vector_data,
            "description": description,
            "utility": utility_score,
            "timestamp": time.time(),
            "generation": len(self.history) + 1
        }
        
        self.memes[meme_id] = entry
        self.history.append(f"Added {concept_name} (Utility: {utility_score:.2f})")
        
        logger.info(f"💡 New meme contributed to culture: {concept_name}")
        self._save()

    def retrieve_meme(self, query_name: str) -> Optional[Dict[str, Any]]:
        """
        名前でミームを検索する（簡易実装）。
        実際にはベクトル類似度検索などが望ましいが、ここではキーワード一致を使用。
        """
        # 完全一致検索
        for mid, data in self.memes.items():
            if data["name"] == query_name:
                return data
        
        # 部分一致検索
        for mid, data in self.memes.items():
            if query_name in data["name"]:
                return data
                
        return None

    def get_top_memes(self, k: int = 5) -> List[Dict[str, Any]]:
        """有用性の高い上位k個のミームを返す"""
        sorted_memes = sorted(self.memes.values(), key=lambda x: x["utility"], reverse=True)
        return sorted_memes[:k]

    def _save(self):
        """データベースをファイルに保存"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.memes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save culture DB: {e}")

    def _load(self):
        """データベースをファイルから読み込み"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.memes = json.load(f)
            except Exception:
                logger.warning("Culture DB file corrupted or empty. Starting fresh.")
                self.memes = {}