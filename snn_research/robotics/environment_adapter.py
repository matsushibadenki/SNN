# ファイルパス: snn_research/robotics/environment_adapter.py
# Title: Environment Adapter (Robotics Integration)
# Description:
#   OpenAI Gym / MuJoCo などの環境とSNNエージェントをつなぐアダプター。
#   スカラー地からスパイクへの変換（エンコーディング）、スパイクからアクションへの変換（デコーディング）を行う。

import gymnasium as gym
import torch
import numpy as np
from typing import Tuple, Any


class EnvironmentAdapter:
    def __init__(self, env_name: str, render: bool = False):
        """
        Args:
            env_name: Gym環境ID
            render: 描画フラグ
        """
        # render_modeの指定 (Gymのバージョンによるが、最近は必須)
        render_mode = "human" if render else None
        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except Exception:
            # Fallback
            self.env = gym.make(env_name)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # メタデータ取得
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_dim = self.env.observation_space.shape[0]
        else:
            self.obs_dim = 1  # Discrete etc.

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.act_dim = self.env.action_space.shape[0]
            self.is_continuous = True
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            self.act_dim = int(self.env.action_space.n)
            self.is_continuous = False
        else:
            # Fallback
            self.act_dim = 1
            self.is_continuous = False

    def reset(self) -> torch.Tensor:
        """
        環境をリセットし、初期観測を返す。
        Returns:
            observation (Tensor): shape (1, obs_dim)
        """
        obs, _ = self.env.reset()
        return self._encode(obs)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        """
        ステップを実行する。
        """
        decoded_action = self._decode(action)
        obs, reward, terminated, truncated, info = self.env.step(
            decoded_action)
        done = terminated or truncated

        return self._encode(obs), float(reward), done, terminated, info

    def _encode(self, observation: np.ndarray) -> torch.Tensor:
        """
        観測(numpy)をTensorに変換。必要に応じてPoplation Codingなどを行う。
        """
        # Trivial implementation: Just tensorize
        # 将来的にはここでRate Codingを行う
        return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    def _decode(self, action: torch.Tensor) -> Any:
        """
        アクションTensorを環境用フォーマットに変換。
        """
        if self.is_continuous:
            # Continuous: Tensor -> numpy array
            return action.detach().cpu().numpy().flatten()
        else:
            # Discrete: Tensor -> int
            # もしactionが確率分布(logits)なら、argmaxを取るなどの処理が必要だが、
            # ここでは既にAction ID (scalar or 1-hot) が来ていると仮定
            if action.numel() > 1:
                # One-hot or logits -> index
                return action.argmax().item()
            return int(action.item())

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
