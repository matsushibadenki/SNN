# /snn_research/agent/active_inference_agent.py
# 日本語タイトル: 能動的推論エージェント (完全整合版)
# 目的: BaseAgentを継承し、自由エネルギー原理に基づき行動を選択する。

import torch
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from snn_research.core.snn_core import SNNCore

class ActiveInferenceAgent(BaseAgent):
    """
    内部モデル(SNN)の予測と外部観測の不一致を最小化するように行動するエージェント。
    """
    def __init__(self, snn_core: SNNCore, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.snn_core = snn_core

    def step(self, observation: torch.Tensor) -> Any:
        """
        観測に基づく推論と行動選択。
        """
        with torch.no_grad():
            # 1. 内部モデルによる推論
            prediction = self.snn_core.forward(observation)
            
            # 2. 内部状態（発火率）の取得
            # SNNCore.get_firing_rates() が辞書を返すことを保証
            firing_rates = self.snn_core.get_firing_rates()
            
            # 3. 驚き（Surprise/Uncertainty）の計算
            # 発火の総和を活動エネルギーの指標として使用
            surprise = sum(firing_rates.values()) if firing_rates else 0.0
            
            # 4. 行動決定
            action = self._select_action_from_prediction(prediction, surprise)
            
            return action

    def _select_action_from_prediction(self, prediction: torch.Tensor, surprise: float) -> Any:
        """
        予測分布と不確実性に基づく行動選択。
        """
        # 単純な最大確率選択
        if prediction.ndim > 1:
            return prediction.argmax(dim=-1)
        return prediction.argmax()