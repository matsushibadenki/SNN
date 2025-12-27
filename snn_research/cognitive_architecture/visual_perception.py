# /snn_research/cognitive_architecture/visual_perception.py
# 日本語タイトル: 視覚知覚モジュール (型安全リセット版)
# 目的: projectorのメソッド呼び出しにおけるCallable判定を追加し、mypyエラーを解消。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class VisualCortex(nn.Module):
    def __init__(self, num_neurons: int = 784, feature_dim: int = 256):
        super().__init__()
        self.num_neurons = num_neurons
        # 例としてprojectorが定義されていると想定
        self.projector: Any = nn.Linear(num_neurons, feature_dim)

    def perceive(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.projector(x)
        return {"features": features}

    def reset_state(self) -> None:
        """状態のリセット。"""
        # [修正] projectorがreset_stateメソッドを持っているか動的に確認
        if hasattr(self.projector, 'reset_state'):
            method = getattr(self.projector, 'reset_state')
            if callable(method):
                method()