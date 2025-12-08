# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# Title: Intrinsic Motivation System (Learning Progress based Curiosity)
# Description:
#   エージェントの内部状態（好奇心、自信、退屈）を管理するシステム。
#   改善点:
#   - 単純な予測誤差ではなく、「学習進捗（Learning Progress）」、すなわち
#     予測誤差の減少率（微分値）を好奇心の主要因とする。
#   - これにより、ランダムなノイズ（学習不能）には興味を示さず、
#     「学習可能な未知」に対して最も強く動機づけられるようになる。

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Deque

class IntrinsicMotivationSystem:
    """
    エージェントの内部状態（好奇心、自信、退屈）と、その源泉を管理するシステム。
    Intelligent Adaptive Curiosity (IAC) の簡易実装を含む。
    """
    def __init__(self, history_length: int = 100):
        # 予測誤差の履歴
        self.prediction_errors: Deque[float] = deque(maxlen=history_length)
        # タスク成功率の履歴
        self.task_success_rates: Deque[float] = deque(maxlen=history_length)
        # タスク類似度の履歴
        self.task_similarities: Deque[float] = deque(maxlen=history_length)
        # 損失の履歴
        self.loss_history: Deque[float] = deque(maxlen=history_length)
        
        # 学習進捗（LP）の履歴: LP = Error(t-1) - Error(t)
        self.learning_progress: Deque[float] = deque(maxlen=history_length)
        
        self.curiosity_context: Optional[Any] = None
        self.max_learning_progress: float = 0.0

    def update_metrics(self, prediction_error: float, success_rate: float, task_similarity: float, loss: float, context: Optional[Any] = None):
        """
        最新のタスク実行結果から各メトリクスを更新する。
        """
        # 前回の予測誤差を取得（LP計算用）
        prev_error = self.prediction_errors[-1] if self.prediction_errors else prediction_error

        self.prediction_errors.append(prediction_error)
        self.task_success_rates.append(success_rate)
        self.task_similarities.append(task_similarity)
        self.loss_history.append(loss)
        
        # 学習進捗の計算 (平滑化のため移動平均との差分をとるのが一般的だが、ここでは簡易的に直近差分)
        # 正の値が大きいほど「理解が進んでいる」状態
        lp = max(0.0, prev_error - prediction_error)
        self.learning_progress.append(lp)

        # 最も「学びがいのある」コンテキストを記録
        if lp > self.max_learning_progress:
            self.max_learning_progress = lp
            self.curiosity_context = context
            # print(f"🌟 Learning Breakthrough! High Learning Progress ({lp:.4f}) in context: {str(context)[:50]}")

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
        好奇心を計算する。
        以前の実装: 予測誤差の平均。
        新しい実装: 学習進捗(LP)の平均。LPが高い（＝急速に学んでいる）ほど好奇心が高まる。
        """
        if not self.learning_progress:
            return 0.5
        
        # 直近の学習進捗の平均
        # ノイズによる変動を抑えるため、少し長めのウィンドウで平均を取る
        recent_lp = list(self.learning_progress)[-10:]
        avg_lp = float(np.mean(recent_lp))
        
        # 0.0 ~ 1.0 にスケーリング（適応的）
        # 期待されるLPの最大値を動的に更新してもよいが、ここでは固定スケール
        normalized_curiosity = min(1.0, avg_lp * 10.0) 
        
        return normalized_curiosity

    def _calculate_confidence(self) -> float:
        """
        自信を計算する。タスクの成功率の平均値として定義。
        """
        if not self.task_success_rates:
            return 0.5
        return float(np.mean(self.task_success_rates))

    def _calculate_boredom(self) -> float:
        """
        退屈を計算する。
        学習進捗が停滞している（LPが低い）かつ、タスクが類似している場合に退屈する。
        """
        if len(self.learning_progress) < 10 or not self.task_similarities:
            return 0.0

        # 学習進捗の低下（停滞）
        avg_lp = float(np.mean(list(self.learning_progress)[-10:]))
        
        # LPが低いほど停滞度が高い
        stagnation = 1.0 - min(1.0, avg_lp * 10.0)
        
        avg_similarity = float(np.mean(self.task_similarities))
        
        # 似たようなタスクで、かつ何も学べていない時に退屈が最大化する
        return stagnation * avg_similarity
