# ファイルパス: snn_research/social/theory_of_mind.py
# Title: Theory of Mind (ToM) Module v1.0
# Description:
#   エージェントが他者の行動意図や信念を推定するためのモジュール。
#   相互作用履歴に基づき、相手の次の行動（投票、協力など）を予測する。

import torch
import torch.nn as nn
from typing import Dict, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TheoryOfMindModule(nn.Module):
    """
    簡易的な心の理論（ToM）モジュール。
    相手のIDと過去の行動から、メンタルモデル（内部状態の推定）を構築する。
    """

    def __init__(self, observation_dim: int = 10, hidden_dim: int = 32, history_len: int = 5):
        super().__init__()
        self.history_len = history_len
        self.observation_dim = observation_dim

        # メンタルモデル用SNN (簡易的なMLP/RNNとして実装)
        # 入力: [history_len * observation_dim] -> 出力: [action_prob]
        self.predictor = nn.Sequential(
            nn.Linear(history_len * observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 0.0 (Reject) ~ 1.0 (Approve/Cooperate)
            nn.Sigmoid()
        )

        # エージェントごとの履歴: AgentID -> Deque[Observation]
        self.interaction_history: Dict[str, Deque[torch.Tensor]] = {}

        logger.info("🧠 TheoryOfMindModule initialized.")

    def observe_agent(self, agent_id: str, action_vector: torch.Tensor):
        """
        エージェントの行動を観察し、履歴に追加する。
        action_vector: 行動の特徴量 (例: 投票内容、発言内容の埋め込み)
        """
        if agent_id not in self.interaction_history:
            self.interaction_history[agent_id] = deque(maxlen=self.history_len)

        # パディング処理
        if action_vector.shape[0] < self.observation_dim:
            padded = torch.zeros(self.observation_dim)
            padded[:action_vector.shape[0]] = action_vector
            action_vector = padded

        self.interaction_history[agent_id].append(action_vector)

    def predict_action(self, agent_id: str) -> float:
        """
        特定のエージェントの次の行動（例えば、賛成確率）を予測する。
        """
        if agent_id not in self.interaction_history or len(self.interaction_history[agent_id]) < 1:
            return 0.5  # 情報なし

        history = list(self.interaction_history[agent_id])
        # 足りない分はゼロパディング
        while len(history) < self.history_len:
            history.insert(0, torch.zeros(self.observation_dim))

        input_tensor = torch.cat(history).unsqueeze(
            0)  # [1, history_len * obs_dim]

        with torch.no_grad():
            prediction = self.predictor(input_tensor).item()

        return prediction

    def update_model(self, agent_id: str, actual_outcome: float):
        """
        予測と実際の結果との誤差に基づいてメンタルモデルを更新する（オンライン学習）。
        """
        # (簡易実装のため省略。本来はここでpredictorの逆伝播を行う)
        pass
