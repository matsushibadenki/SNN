# /snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: グローバル・ワークスペース (完全整合版)
# 目的: 認知アーキテクチャ全体の中央ハブとして、情報のアップロード、放送、取得を担う。

from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING, Deque
from collections import deque
import operator
import logging

if TYPE_CHECKING:
    from snn_research.distillation.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class AttentionHub:
    """
    Winner-Take-All競合により、最も重要な情報を選択する注意メカニズム。
    """
    def __init__(self, inhibition_strength: float = 0.5, decay_rate: float = 0.1):
        self.history: List[str] = []
        self.inhibition_scores: Dict[str, float] = {}
        self.inhibition_strength = inhibition_strength
        self.decay_rate = decay_rate

    def select_winner(self, salience_signals: Dict[str, float]) -> Optional[str]:
        if not salience_signals:
            return None

        # 抑制の減衰（順応）
        for key in list(self.inhibition_scores.keys()):
            self.inhibition_scores[key] *= (1.0 - self.decay_rate)
            if self.inhibition_scores[key] < 0.01:
                del self.inhibition_scores[key]

        # 実効顕著性の計算
        adjusted_signals: Dict[str, float] = {
            name: max(0.0, raw - self.inhibition_scores.get(name, 0.0))
            for name, raw in salience_signals.items()
        }

        if not adjusted_signals or max(adjusted_signals.values()) <= 0:
            return None
            
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        
        # 勝者への抑制を強化（注意の交代を促す）
        self.inhibition_scores[winner] = self.inhibition_scores.get(winner, 0.0) + self.inhibition_strength
        
        self.history.append(winner)
        if len(self.history) > 20:
            self.history.pop(0)

        return winner


class GlobalWorkspace:
    """
    意識の放送、情報の統合、およびワーキングメモリを管理するハブ。
    """
    def __init__(self, capacity: int = 7, model_registry: Optional["ModelRegistry"] = None):
        self.blackboard: Dict[str, Any] = {} 
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.attention_hub = AttentionHub()
        
        self.model_registry = model_registry
        
        self.conscious_broadcast_content: Optional[Any] = None
        self.current_context_source: Optional[str] = None

        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.working_memory: List[Any] = []
        
        logger.info("✨ Global Workspace (Consciousness Hub) initialized.")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """放送内容を受け取るモジュールを登録する。"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        """[mypy修正] 各モジュールから情報を書き込むメインインターフェース。"""
        self.blackboard[source] = {"data": data, "salience": salience}

    def add_content(self, source: str, data: Any, salience: float = 1.0):
        """ArtificialBrain用のエイリアス。"""
        self.upload_to_workspace(source, data, salience)

    def get_information(self, source: str) -> Any:
        """[mypy修正] 特定のソースから現在のデータを取得する。"""
        info = self.blackboard.get(source)
        return info['data'] if info else None

    def get_summary(self) -> List[Dict[str, Any]]:
        """全情報の要約を取得する。"""
        return [
            {"source": k, "data": v["data"], "salience": v["salience"]}
            for k, v in self.blackboard.items()
        ]

    def conscious_broadcast_cycle(self):
        """1サイクルの注意選択と放送を実行する。"""
        if not self.blackboard:
            self.conscious_broadcast_content = None
            return

        salience_signals = {
            source: info["salience"]
            for source, info in self.blackboard.items()
        }

        winner = self.attention_hub.select_winner(salience_signals)

        if winner and winner in self.blackboard:
            winner_info = self.blackboard[winner]
            content = winner_info['data']
            self.conscious_broadcast_content = content
            self.current_context_source = winner
            
            # メタデータの付与
            broadcast_packet = content
            if isinstance(broadcast_packet, dict):
                broadcast_packet = broadcast_packet.copy()
                broadcast_packet["source_module"] = winner
                broadcast_packet["salience"] = winner_info["salience"]
                self.stream_of_consciousness.append(broadcast_packet)

            logger.info(f"📡 CONSCIOUS BROADCAST: <{winner}> took the stage")
            self._notify_subscribers(winner, broadcast_packet)
        else:
            self.conscious_broadcast_content = None

        # 放送後にブラックボードをリセット（GWTの揮発性モデル）
        self.blackboard.clear()

    def _notify_subscribers(self, source: str, content: Any):
        for callback in self.subscribers:
            try:
                callback(source, content)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    def broadcast(self):
        """エイリアスメソッド。"""
        self.conscious_broadcast_cycle()
