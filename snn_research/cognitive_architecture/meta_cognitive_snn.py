# snn_research/cognitive_architecture/meta_cognitive_snn.py
# メタ認知SNN
# 概要：システム自身の学習プロセスやパフォーマンスを監視・評価する。
import numpy as np
from collections import deque

class MetaCognitiveSNN:
    """
    学習プロセスのメタデータを監視し、システムのパフォーマンスを評価する。
    これにより、より高度な自己改善の意思決定を可能にする。
    """
    def __init__(self, window_size=50):
        """
        Args:
            window_size (int): 評価に使用するデータウィンドウのサイズ。
        """
        self.loss_history = deque(maxlen=window_size)
        self.computation_time_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)

    def update_metadata(self, loss, computation_time, accuracy):
        """
        最新の学習ステップからメタデータを更新する。

        Args:
            loss (float): 損失関数の値。
            computation_time (float): 計算時間（秒）。
            accuracy (float): 精度。
        """
        self.loss_history.append(loss)
        self.computation_time_history.append(computation_time)
        self.accuracy_history.append(accuracy)

    def evaluate_performance(self):
        """
        蓄積されたメタデータから現在のパフォーマンスを診断する。

        Returns:
            dict: パフォーマンス診断結果を含む辞書。
                   (e.g., "knowledge_gap", "capability_gap", "optimized")
        """
        if len(self.loss_history) < 2:
            return {"status": "initializing", "details": "Not enough data for evaluation."}

        # 1. 学習の収束速度を評価
        loss_gradient = np.gradient(list(self.loss_history))
        convergence_speed = np.mean(loss_gradient[-10:]) # 直近10ステップの勾配平均

        # 2. 精度の停滞を評価
        accuracy_change = np.mean(np.diff(list(self.accuracy_history))) if len(self.accuracy_history) > 1 else 0

        # 診断ロジック
        if abs(convergence_speed) > 0.01:
             # 学習は順調に進んでいる
            return {"status": "learning", "convergence_speed": convergence_speed}
        
        elif np.mean(self.accuracy_history) < 0.5 and abs(accuracy_change) < 0.001:
             # 精度が低いまま学習が停滞している -> 知識不足の可能性
            return {"status": "knowledge_gap", "details": "Accuracy is low and learning has plateaued. Consider acquiring new data."}
        
        elif np.mean(self.accuracy_history) >= 0.5 and abs(accuracy_change) < 0.001:
            # 精度はそこそこだが停滞している -> モデルの能力不足の可能性
            return {"status": "capability_gap", "details": "Learning has converged but accuracy may be improved. Consider evolving the model architecture."}
        
        else:
            return {"status": "optimized", "details": "Performance is stable."}