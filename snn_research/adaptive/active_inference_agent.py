# ファイルパス: snn_research/adaptive/active_inference_agent.py
# Title: Active Inference Agent (Free Energy Principle)
# Description:
#   自由エネルギー原理（FEP）に基づく能動的推論エージェント。
#   予測誤差（変分自由エネルギー）を最小化するように行動を選択し、内部モデルを更新する。

import torch
import torch.nn as nn
from typing import Tuple


class ActiveInferenceAgent(nn.Module):
    """
    能動的推論 (Active Inference) を行うエージェント。

    仕組み:
    1. Perception (知覚): 感覚入力から状態を推定し、予測誤差（Surprise）を計算。
    2. Action (行動): 予測誤差（期待自由エネルギー）を最小化する行動を選択。
    3. Learning (学習): 内部モデル（生成モデル）を更新して予測精度を向上。
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 0.01
    ):
        super().__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # --- Generative Model (Internal Model) ---
        # 状態予測モデル: State(t) -> State(t+1)
        self.transition_model = nn.Linear(state_dim + action_dim, state_dim)

        # 観測予測モデル: State(t) -> Observation(t)
        self.observation_model = nn.Linear(state_dim, observation_dim)

        # 変分パラメータ (信念状態)
        # 現在の状態推定値 mu
        self.mu = nn.Parameter(torch.randn(1, state_dim), requires_grad=True)

        # 最適化用
        self.optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()) +
            list(self.observation_model.parameters()),
            lr=learning_rate
        )

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        1ステップの能動的推論サイクルを実行。

        Args:
            observation: 現在の感覚入力 (Batch, ObsDim)
            prev_action: 直前の行動 (Batch, ActDim)

        Returns:
            action: 選択された次の行動 (Batch, ActDim)
            free_energy: 現在の自由エネルギー (Surprise)
        """
        # 1. Perception Step (状態推定)
        # 観測モデルによる予測
        predicted_obs = self.observation_model(self.mu)

        # 予測誤差 (Prediction Error)
        # 単純化のため、ガウス分布を仮定した二乗誤差として計算
        prediction_error = observation - predicted_obs
        variational_free_energy = 0.5 * torch.sum(prediction_error ** 2)

        # 内部状態の更新 (勾配降下法によるFEP最小化)
        # 実際にはmuに対する勾配を計算して更新するが、ここでは簡易的にモデル学習と同時に行う
        pass

        # 2. Action Selection (行動選択)
        # 期待自由エネルギー (EFE) を最小化する行動を探索
        # ここでは簡易的に、現在の状態からランダムにサンプリングした行動候補の中から
        # 最も予測誤差を小さくする（または望ましい状態に近づく）ものを選ぶ

        # サンプリング数
        num_samples = 10
        sampled_actions = torch.randn(num_samples, self.action_dim)
        if prev_action.device.type == 'cuda':
            sampled_actions = sampled_actions.cuda()

        best_action = None
        min_efe = float('inf')

        # 本来は未来への展開が必要だが、ここでは1ステップ先のみ考える
        current_state = self.mu.detach()

        for i in range(num_samples):
            action_candidate = sampled_actions[i:i+1]

            # 遷移モデルで次の状態を予測
            input_tensor = torch.cat([current_state, action_candidate], dim=1)
            predicted_next_state = self.transition_model(input_tensor)

            # 期待自由エネルギー (EFE) の簡易計算
            # EFE ≈ Ambiguity + Risk (Divergence from verify preferred inputs)
            # ここでは簡単のため、予測された次の状態が「安定しているか」(ノルムが小さいか) を指標とする
            # (実際にはゴール状態との距離などを使う)
            efe = torch.norm(predicted_next_state)

            if efe < min_efe:
                min_efe = efe
                best_action = action_candidate

        if best_action is None:
            best_action = torch.zeros(1, self.action_dim)

        return best_action, variational_free_energy.item()

    def update_model(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor):
        """
        内部モデルの学習。
        """
        self.optimizer.zero_grad()

        # 1. 状態予測の誤差
        current_input = torch.cat([self.mu.detach(), action], dim=1)
        predicted_next_state = self.transition_model(current_input)

        # 次のステップの観測予測
        predicted_next_obs = self.observation_model(predicted_next_state)

        # 損失: 観測の再構成誤差
        loss = nn.MSELoss()(predicted_next_obs, next_observation)

        loss.backward()
        self.optimizer.step()

        # 信念状態 (mu) の更新 (単純な移動平均)
        with torch.no_grad():
            # 観測から逆算した状態 (理想) に近づける... としたいが、
            # ここでは遷移モデルの予測先に少し近づける
            self.mu.data = 0.9 * self.mu.data + 0.1 * predicted_next_state.data
