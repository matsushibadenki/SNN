# ファイルパス: snn_research/core/networks/abstract_snn_network.py
# Title: 抽象SNNネットワーク (Recursion Fix)
# Description: reset_stateにおける無限再帰バグを修正。

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

# 絶対インポート
from snn_research.core.learning_rule import AbstractLearningRule

class AbstractSNNNetwork(nn.Module, ABC):
    """
    局所学習規則を適用するための抽象基底クラス。
    """
    learning_rules: List[AbstractLearningRule]

    def __init__(self) -> None:
        super().__init__()
        self.learning_rules = []
        # 各層の内部状態（膜電位、スパイク、入力電流など）を保持する辞書
        # Key: "layer_name_attribute" (例: "fc1_pre_activity")
        self.model_state: Dict[str, torch.Tensor] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播。model_state の更新もここで行うことを想定。"""
        pass

    def add_learning_rule(self, rule: AbstractLearningRule) -> None:
        """学習規則を追加する。"""
        self.learning_rules.append(rule)

    def run_learning_step(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        登録されたすべての学習規則に対して1ステップの更新を実行する。
        """
        stats = {}
        for rule in self.learning_rules:
            # 各ルールに現在の model_state を渡して更新
            rule_stats = rule.step(inputs, targets, self.model_state)
            
            # レイヤー名が紐付いている場合はプレフィックスとして使用
            prefix = rule.layer_name + "_" if rule.layer_name else ""
            
            for k, v in rule_stats.items():
                stats[f"{prefix}{k}"] = v
        return stats

    def reset_state(self) -> None:
        """
        ネットワークの内部状態（膜電位など）をリセットする。
        """
        self.model_state = {}
        
        # [Fix] RecursionError回避: self.modules() は self を含むため、無限再帰を引き起こす。
        # self 自身の場合はスキップする。
        for m in self.modules():
            if m is self:
                continue
            
            if hasattr(m, 'reset'):
                m.reset() # type: ignore
            elif hasattr(m, 'reset_state'):
                m.reset_state() # type: ignore