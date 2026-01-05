# snn_research/rl_env/grid_world.py
# (新規作成)
# Title: Grid World 環境
# Description: ROADMAPフェーズ2「階層的因果学習」の検証のため、
#              複数ステップの行動選択を必要とするシンプルな迷路探索タスクを提供する。

import torch
from typing import Tuple

class GridWorldEnv:
    """
    エージェントがゴールを目指す、シンプルなグリッドワールド環境。
    """
    def __init__(self, size: int = 5, max_steps: int = 50, device: str = 'cpu'):
        self.size = size
        self.max_steps = max_steps
        self.device = device
        
        self.agent_pos = torch.zeros(2, device=self.device, dtype=torch.long)
        self.goal_pos = torch.zeros(2, device=self.device, dtype=torch.long)
        
        self.current_step = 0
        self.reset()

    def _get_state(self) -> torch.Tensor:
        """現在の状態（エージェント位置とゴール位置）をベクトルとして返す。"""
        # 状態を正規化して [-1, 1] の範囲にする
        state = torch.cat([
            (self.agent_pos / (self.size - 1)) * 2 - 1,
            (self.goal_pos / (self.size - 1)) * 2 - 1
        ]).float()
        return state

    def reset(self) -> torch.Tensor:
        """環境をリセットし、新しいエージェントとゴールの位置を設定する。"""
        self.agent_pos = torch.randint(0, self.size, (2,), device=self.device)
        self.goal_pos = torch.randint(0, self.size, (2,), device=self.device)
        # エージェントとゴールが同じ場所から始まらないようにする
        while torch.equal(self.agent_pos, self.goal_pos):
            self.goal_pos = torch.randint(0, self.size, (2,), device=self.device)
        
        self.current_step = 0
        
        # print(f"🌍 New Grid World: Agent at {self.agent_pos.cpu().numpy()}, Goal at {self.goal_pos.cpu().numpy()}")
        return self._get_state()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        エージェントの行動を受け取り、(次の状態, 報酬, 完了フラグ) を返す。

        Args:
            action (int): 0:上, 1:下, 2:左, 3:右

        Returns:
            Tuple[torch.Tensor, float, bool]: (次の状態, 報酬, 完了フラグ)。
        """
        self.current_step += 1

        # 行動に基づいてエージェントを移動
        if action == 0: # 上
            self.agent_pos[1] += 1
        elif action == 1: # 下
            self.agent_pos[1] -= 1
        elif action == 2: # 左
            self.agent_pos[0] -= 1
        elif action == 3: # 右
            self.agent_pos[0] += 1
        
        # グリッドの境界内に収める
        self.agent_pos = torch.clamp(self.agent_pos, 0, self.size - 1)

        # 報酬の計算
        if torch.equal(self.agent_pos, self.goal_pos):
            reward = 1.0  # ゴールに到達
            done = True
        else:
            reward = -0.05  # 移動コスト
            done = False

        # 最大ステップ数に達したら終了
        if self.current_step >= self.max_steps:
            done = True
        
        next_state = self._get_state()

        return next_state, reward, done