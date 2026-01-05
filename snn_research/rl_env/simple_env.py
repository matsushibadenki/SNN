# matsushibadenki/snn4/snn_research/rl_env/simple_env.py
# Title: シンプルな強化学習環境
# Description: 生物学的学習則を持つエージェントが学習するための、
#              基本的な強化学習タスク（パターンマッチング）を提供します。

import torch
from typing import Tuple

class SimpleEnvironment:
    """
    エージェントが目標パターンを当てるシンプルな強化学習環境。
    """
    def __init__(self, pattern_size: int, device: str = 'cpu'):
        self.pattern_size = pattern_size
        self.device = device
        self.target_pattern = torch.zeros(pattern_size, device=self.device)
        self.reset()

    def reset(self) -> torch.Tensor:
        """環境をリセットし、新しい目標パターンを生成する。"""
        self.target_pattern = (torch.rand(self.pattern_size, device=self.device) > 0.5).float()
        print(f"🌍 New Target Pattern: {self.target_pattern.cpu().numpy().astype(int)}")
        return self.target_pattern

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        エージェントの行動を受け取り、報酬と次の状態を返す。

        Args:
            action (torch.Tensor): エージェントの出力（発火パターン）。

        Returns:
            Tuple[torch.Tensor, float, bool]: (次の状態, 報酬, 完了フラグ)。
        """
        # 報酬の計算: 目標パターンとエージェントの行動がどれだけ一致しているか
        correct_matches = (action == self.target_pattern).float().sum()
        reward = (correct_matches / self.pattern_size).item()

        # 完了フラグ (このシンプルな環境では常に1ステップで完了)
        done = True
        
        # 新しいタスクを開始するために環境をリセット
        next_state = self.reset()

        return next_state, reward, done
