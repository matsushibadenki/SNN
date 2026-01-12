# snn_research/learning_rules/probabilistic_hebbian.py
# Title: Probabilistic Hebbian Learning Rule
# Description:
#   確率的スパイクニューロン向けのヘブ学習則。
#   BioLearningRuleインターフェースに準拠。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_rule import BioLearningRule

class ProbabilisticHebbian(nn.Module, BioLearningRule):
    """
    確率的スパイクニューロンのためのシンプルなヘブ学習則。
    シナプス前後のニューロンが同時に(確率的に)活動した場合に結合を強化する。
    """
    def __init__(self, learning_rate: float = 0.005, weight_decay: float = 0.0001):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # print("💡 Probabilistic Hebbian learning rule initialized.") # ログ抑制

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

        Returns:
            (dw, backward_credit)
        """
        # バッチ次元の処理: (Batch, N) -> (Batch, N)
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)

        # ヘブ則: 同時活動による結合強化項
        # (Batch, N_post, 1) * (Batch, 1, N_pre) -> (Batch, N_post, N_pre)
        hebbian_term = torch.bmm(post_spikes.unsqueeze(2), pre_spikes.unsqueeze(1))
        
        # バッチ平均をとって勾配とする
        mean_hebbian = hebbian_term.mean(dim=0)

        # 重み減衰項 (過剰な強化を防ぎ、安定させる)
        decay_term = self.weight_decay * weights

        # 重み変化量
        dw = self.learning_rate * (mean_hebbian - decay_term)

        # この学習則は局所的なので、逆方向のクレジット信号は生成しない
        backward_credit = None

        return dw, backward_credit