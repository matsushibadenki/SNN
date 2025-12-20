# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
from typing import Dict, Any, List, Callable, Optional, Tuple, TYPE_CHECKING, Deque
from collections import deque
import operator
import logging

if TYPE_CHECKING:
    from snn_research.distillation.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# WorkspaceItemクラスが定義されていない場合は定義が必要ですが、
# エラーログに出ていないため、ここでは内部クラスか辞書として扱われている前提で進めます。
# ただし、ArtificialBrain内で使われているため、簡易的なデータクラスとして定義があると安全です。
# ここではエラー箇所の修正に集中します。

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

        # 抑制の減衰
        for key in list(self.inhibition_scores.keys()):
            self.inhibition_scores[key] *= (1.0 - self.decay_rate)
            if self.inhibition_scores[key] < 0.01:
                del self.inhibition_scores[key]

        # 実効顕著性の計算
        adjusted_signals: Dict[str, float] = {}
        for name, raw_salience in salience_signals.items():
            inhibition = self.inhibition_scores.get(name, 0.0)
            adjusted_signals[name] = max(0.0, raw_salience - inhibition)

        if not adjusted_signals:
            return None
            
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        
        self.inhibition_scores[winner] = self.inhibition_scores.get(winner, 0.0) + self.inhibition_strength
        
        self.history.append(winner)
        if len(self.history) > 20:
            self.history.pop(0)

        return winner


class GlobalWorkspace:
    """
    認知アーキテクチャ全体で情報を共有する中央情報ハブ。
    """
    def __init__(self, capacity: int = 7, model_registry: Optional["ModelRegistry"] = None):
        self.blackboard: Dict[str, Any] = {} 
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.attention_hub = AttentionHub()
        
        self.model_registry = model_registry
        
        self.conscious_broadcast_content: Optional[Any] = None
        self.current_context_source: Optional[str] = None

        # [Fix 1] 型アノテーションを追加
        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.working_memory: List[Any] = [] # 追加: ArtificialBrainやDashboardで参照されている可能性があるため
        
        logger.info("✨ Global Workspace (Consciousness Hub) initialized.")

    def subscribe(self, callback: Callable[[str, Any], None]):
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        self.blackboard[source] = {"data": data, "salience": salience}

    def conscious_broadcast_cycle(self):
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
            self.conscious_broadcast_content = winner_info['data']
            self.current_context_source = winner
            
            broadcast_packet = self.conscious_broadcast_content
            if isinstance(broadcast_packet, dict):
                broadcast_packet = broadcast_packet.copy()
                broadcast_packet["source_module"] = winner

            # [Fix 2] unique_itemsの型定義 (元コードにはなかったが、整合性のため)
            # ここでは元コードのロジックを維持しつつ、必要なら辞書を用意
            
            # 履歴に追加
            if isinstance(broadcast_packet, dict):
                self.stream_of_consciousness.append(broadcast_packet)

            logger.info(f"📡 CONSCIOUS BROADCAST: <{winner}> took the stage (Salience: {winner_info['salience']:.2f})")
            self._notify_subscribers(winner, broadcast_packet)
        else:
            self.conscious_broadcast_content = None

        self.blackboard.clear()

    def _notify_subscribers(self, source: str, content: Any):
        for callback in self.subscribers:
            try:
                callback(source, content)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}", exc_info=True)

    def get_information(self, source: str) -> Any:
        info = self.blackboard.get(source)
        return info['data'] if info else None