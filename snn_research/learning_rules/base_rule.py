# snn_research/learning_rules/base_rule.py
# Title: 学習ルールの抽象基底クラス
# Description: 全ての学習ルールクラスが継承すべき基本構造を定義します。
# 修正: 階層的因果学習のため、戻り値に逆方向クレジット信号を追加。

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional, Tuple

class BioLearningRule(ABC):
    """生物学的学習ルールのための抽象基底クラス。"""

    @abstractmethod
    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        シナプス重みの変化量を計算する。

        Args:
            (省略)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                (計算された重み変化量 (dw), 前段の層へ伝えるクレジット信号)
        """
        pass