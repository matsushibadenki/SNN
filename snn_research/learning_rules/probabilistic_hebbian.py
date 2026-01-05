# ファイルパス: snn_research/learning_rules/probabilistic_hebbian.py
# (新規作成)
# Title: 確率的ヘブ学習則
# Description: 論文 arXiv:2509.26507v1 のアイデアに基づき、
#              確率的スパイクニューロン向けのヘブ学習則を実装する。

import torch
from typing import Dict, Any, Optional, Tuple
from .base_rule import BioLearningRule

class ProbabilisticHebbian(BioLearningRule):
    """
    確率的スパイクニューロンのためのシンプルなヘブ学習則。
    シナプス前後のニューロンが同時に(確率的に)活動した場合に結合を強化する。
    """
    def __init__(self, learning_rate: float = 0.005, weight_decay: float = 0.0001):
        """
        Args:
            learning_rate (float): 学習率。
            weight_decay (float): 重みの減衰率（安定化のため）。
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        print("💡 Probabilistic Hebbian learning rule initialized.")

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ヘブ則に基づいて重み変化量を計算する。
        dw = lr * (post_spikes * pre_spikes^T - decay * weights)

        Args:
            pre_spikes (torch.Tensor): シナプス前ニューロンのスパイク (確率的な0 or 1)。 (N_pre,)
            post_spikes (torch.Tensor): シナプス後ニューロンのスパイク (確率的な0 or 1)。(N_post,)
            weights (torch.Tensor): 現在の重み行列。(N_post, N_pre)
            optional_params (Optional[Dict[str, Any]]): 追加パラメータ (ここでは未使用)。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                (計算された重み変化量 (dw), 逆方向クレジット信号 (None))
        """
        # ヘブ則: 同時活動による結合強化項
        # torch.outer(post_spikes, pre_spikes) は (N_post, N_pre) の行列を生成
        hebbian_term = torch.outer(post_spikes, pre_spikes)

        # 重み減衰項 (過剰な強化を防ぎ、安定させる)
        decay_term = self.weight_decay * weights

        # 重み変化量
        dw = self.learning_rate * (hebbian_term - decay_term)

        # この学習則は局所的なので、逆方向のクレジット信号は生成しない
        backward_credit = None

        return dw, backward_credit