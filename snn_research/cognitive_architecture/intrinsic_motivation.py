# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# (修正)
# mypyエラー[return-value]を解消するため、numpy.mean()の戻り値をfloat()でキャスト。

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Deque

class IntrinsicMotivationSystem:
    """
    エージェントの内部状態（好奇心、自信、退屈）と、その源泉を管理するシステム。
    """
    def __init__(self, history_length: int = 100):
        self.prediction_errors: Deque[float] = deque(maxlen=history_length)
        self.task_success_rates: Deque[float] = deque(maxlen=history_length)
        self.task_similarities: Deque[float] = deque(maxlen=history_length)
        self.loss_history: Deque[float] = deque(maxlen=history_length)
        self.curiosity_context: Optional[Any] = None
        self.max_prediction_error: float = 0.0

    def update_metrics(self, prediction_error: float, success_rate: float, task_similarity: float, loss: float, context: Optional[Any] = None):
        """
        最新のタスク実行結果から各メトリクスを更新する。
        """
        self.prediction_errors.append(prediction_error)
        self.task_success_rates.append(success_rate)
        self.task_similarities.append(task_similarity)
        self.loss_history.append(loss)

        if prediction_error > self.max_prediction_error:
            self.max_prediction_error = prediction_error
            self.curiosity_context = context
            print(f"🌟 新しい好奇心の対象を発見: {str(context)[:100]}")

    def get_internal_state(self) -> Dict[str, Any]:
        """
        現在の内部状態を定量的な指標として計算する。
        """
        state = {
            "curiosity": self._calculate_curiosity(),
            "confidence": self._calculate_confidence(),
            "boredom": self._calculate_boredom(),
            "curiosity_context": self.curiosity_context
        }
        return state

    def _calculate_curiosity(self) -> float:
        """
        好奇心を計算する。予測誤差の平均値として定義。
        """
        if not self.prediction_errors:
            return 0.5
        # --- ▼ 修正 ▼ ---
        return float(np.mean(self.prediction_errors))
        # --- ▲ 修正 ▲ ---

    def _calculate_confidence(self) -> float:
        """
        自信を計算する。タスクの成功率の平均値として定義。
        """
        if not self.task_success_rates:
            return 0.5
        # --- ▼ 修正 ▼ ---
        return float(np.mean(self.task_success_rates))
        # --- ▲ 修正 ▲ ---

    def _calculate_boredom(self) -> float:
        """
        退屈を計算する。学習の停滞度とタスクの類似度から定義。
        """
        if len(self.loss_history) < 2 or not self.task_similarities:
            return 0.0

        loss_change_rate = np.mean(np.abs(np.diff(list(self.loss_history))))
        stagnation = 1.0 - np.tanh(loss_change_rate * 10)
        avg_similarity = np.mean(self.task_similarities)
        
        return stagnation * avg_similarity