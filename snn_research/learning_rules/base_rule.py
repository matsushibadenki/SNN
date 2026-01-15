# snn_research/learning_rules/base_rule.py
# 修正: 引数順序を (pre, post, weights, optional_params) に統一

from abc import ABC, abstractmethod
import torch
from typing import Any, Tuple, Optional, Dict

class BioLearningRule(ABC):
    """
    生物学的妥当性を持つ学習則の抽象基底クラス。
    """
    @abstractmethod
    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        weights: torch.Tensor, 
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Any]:
        """
        シナプス重みを更新する。
        
        Args:
            pre_spikes (Tensor): プレニューロンのスパイク (Batch, N_pre)
            post_spikes (Tensor): ポストニューロンのスパイク (Batch, N_post)
            weights (Tensor): 現在の重み
            optional_params (Dict, optional): 報酬信号などの追加パラメータ
            **kwargs: その他のオプション引数

        Returns:
            (delta_w, info): 重み変化量(Tensor)とその他の情報(Any)のタプル
        """
        pass