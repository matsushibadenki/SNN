# ファイルパス: snn_research/training/rl/replay_buffer.py
# Title: Replay Buffer
# Description: 強化学習用の経験再生バッファ。

import numpy as np
import collections


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        経験を保存する。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        バッチサイズ分の経験をランダムに取得する。
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)
