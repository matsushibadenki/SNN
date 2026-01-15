# snn_research/cognitive_architecture/amygdala.py
# 修正: イリヤ・サツケバーの仮説に基づき、感情を「価値関数(Value Function)」として実装する。
#       意思決定の「暗闇を照らす直感」として機能させる。

import torch
import torch.nn as nn
from typing import Dict, Optional

class Amygdala(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 感覚入力から「情動価（Valence）」と「覚醒度（Arousal）」を予測する
        # これが「価値関数」の本体となる
        self.value_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # 出力: -1.0(Bad) to 1.0(Good)
        )
        
        # 恒常性（Homeostasis）の基準値
        self.base_value = 0.0
        
    def forward(self, sensory_input: torch.Tensor, internal_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        感覚入力に対して、直感的な「価値」を返す。
        これが高いほど、その状態や行動は「生存/目的に適っている」と判断される。
        """
        # 単純なフォワードパスではなく、記憶(Hippocampus)からの文脈も加味するのが理想だが
        # まずは感覚入力からの即時評価（直感）を実装
        
        estimated_value = self.value_estimator(sensory_input)
        
        # 値を -1 ~ 1 にクリップまたは活性化
        value_signal = torch.tanh(estimated_value)
        
        return value_signal

    def update_value_function(self, sensory_input: torch.Tensor, real_reward: float):
        """
        実際の外部報酬が得られたとき、価値観数を更新（学習）する。
        これにより「何が良いことか」の直感を磨く。
        """
        # 簡易的なTD学習または教師あり学習
        predicted_value = self.value_estimator(sensory_input)
        target = torch.tensor([[real_reward]], device=sensory_input.device)
        
        criterion = nn.MSELoss()
        loss = criterion(predicted_value, target)
        
        # ここでBackpropまたは局所学習則を適用
        # （SNNのコンテキストに合わせてHeavysideなどを適用する場合もある）
        return loss