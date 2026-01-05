# /snn_research/agent/base_agent.py
# 日本語タイトル: エージェント基底クラス (BaseAgent)
# 目的: 全てのエージェントの共通インターフェースを定義し、mypyエラーを解消する。

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """
    人工脳プロジェクトにおけるエージェントの抽象基底クラス。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def step(self, observation: torch.Tensor) -> Any:
        """
        1ステップの環境観測を受け取り、行動を返す。
        """
        pass

    def reset(self) -> None:
        """
        エージェントの内部状態をリセットする。
        """
        pass