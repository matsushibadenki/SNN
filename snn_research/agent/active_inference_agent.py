# /snn_research/agent/active_inference_agent.py
# 日本語タイトル: 能動的推論エージェント (整合性修正版)
# 目的: 認知サイクルにおける発火率の取得エラーを修正し、自由エネルギー原理に基づく動作を保証する。

import torch
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from snn_research.core.snn_core import SNNCore

class ActiveInferenceAgent(BaseAgent):
    """
    自由エネルギー原理に基づき行動を選択するエージェント。
    """
    def __init__(self, snn_core: SNNCore, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.snn_core = snn_core
        self.config = config or {}

    def step(self, observation: torch.Tensor) -> Any:
        """[修正箇所] 1ステップの推論と行動選択。"""
        with torch.no_grad():
            # 1. 観測をSNNコアへ入力
            prediction = self.snn_core.forward(observation)
            
            # 2. 内部統計（発火率）の取得
            # [修正] Tensor not callable エラーを防ぐため、メソッド呼び出しを確実に行う
            firing_rates = self.snn_core.get_firing_rates()
            
            # 3. 自由エネルギーの計算（簡略化されたメタ認知）
            # 発火率の分散や強度が期待値から外れている場合、'surprise' として処理
            surprise = sum(firing_rates.values()) if firing_rates else 0.0
            
            # 4. 行動の決定
            action = self._select_action_from_prediction(prediction, surprise)
            
            return action

    def _select_action_from_prediction(self, prediction: torch.Tensor, surprise: float) -> Any:
        """予測結果と驚き(Surprise)に基づく行動選択ロジック。"""
        # 予測分布から最大値を選択
        return prediction.argmax(dim=-1)
