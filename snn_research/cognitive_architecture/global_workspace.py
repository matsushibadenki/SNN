# /snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: グローバル・ワークスペース (機能完備版)
# 目的: モジュール間情報共有のハブとして、意識のストリームとサマリー機能を提供する。

from typing import Dict, Any, List, Callable, Optional, Deque
from collections import deque
import operator
import logging

logger = logging.getLogger(__name__)

class AttentionHub:
    """
    Winner-Take-All競合による注意選択メカニズム。
    """
    def __init__(self, inhibition_strength: float = 0.5, decay_rate: float = 0.1):
        self.inhibition_scores: Dict[str, float] = {}
        self.inhibition_strength = inhibition_strength
        self.decay_rate = decay_rate

    def select_winner(self, salience_signals: Dict[str, float]) -> Optional[str]:
        if not salience_signals:
            return None

        # 抑制の減衰
        for key in list(self.inhibition_scores.keys()):
            self.inhibition_scores[key] *= (1.0 - self.decay_rate)
            if self.inhibition_scores[key] < 0.01:
                del self.inhibition_scores[key]

        # 実効顕著性の計算
        adjusted_signals = {
            name: max(0.0, s - self.inhibition_scores.get(name, 0.0))
            for name, s in salience_signals.items()
        }

        if not adjusted_signals:
            return None
            
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        self.inhibition_scores[winner] = self.inhibition_scores.get(winner, 0.0) + self.inhibition_strength
        return winner

class GlobalWorkspace:
    """
    情報を保持し、全モジュールへ放送する中央情報ハブ。
    """
    def __init__(self, capacity: int = 7):
        self.blackboard: Dict[str, Any] = {} 
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.attention_hub = AttentionHub()
        
        self.conscious_broadcast_content: Optional[Any] = None
        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        logger.info("Global Workspace initialized.")

    def add_content(self, source: str, data: Any, salience: float = 1.0):
        """[新規追加] ワークスペースに情報を書き込む。"""
        self.blackboard[source] = {"data": data, "salience": salience}

    def update(self, source: str, data: Any):
        """エイリアスメソッド。"""
        self.add_content(source, data)

    def get_summary(self) -> List[Dict[str, Any]]:
        """[新規追加] 現在のブラックボードの内容をリスト形式で返す。"""
        summary = []
        for source, info in self.blackboard.items():
            summary.append({
                "source": source,
                "data": info["data"],
                "salience": info["salience"]
            })
        return summary

    def broadcast(self):
        """現在の勝者情報を全サブスクライバーへ通知する。"""
        if not self.blackboard:
            return

        salience_signals = {k: v["salience"] for k, v in self.blackboard.items()}
        winner = self.attention_hub.select_winner(salience_signals)

        if winner:
            content = self.blackboard[winner]["data"]
            self.conscious_broadcast_content = content
            
            packet = {"source": winner, "content": content}
            self.stream_of_consciousness.append(packet)
            
            for callback in self.subscribers:
                callback(winner, content)
        
        # サイクル終了後にクリア（GWTの瞬時放送モデル）
        self.blackboard.clear()

    def subscribe(self, callback: Callable[[str, Any], None]):
        self.subscribers.append(callback)
