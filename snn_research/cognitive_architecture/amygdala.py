# snn_research/cognitive_architecture/amygdala.py
# 修正: processメソッドがテストの期待通り辞書を返すように変更

import torch
import torch.nn as nn
from typing import Optional, Union, Dict

class Amygdala(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.value_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.base_value = 0.0
        
        # テスト用簡易辞書
        self.sentiment_lexicon = {
            "成功": 0.8, "喜び": 0.7, "良い": 0.5,
            "失敗": -0.8, "危険": -0.9, "エラー": -0.7
        }

    def forward(self, sensory_input: torch.Tensor, internal_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        estimated_value = self.value_estimator(sensory_input)
        return torch.tanh(estimated_value)

    def process(self, text: str) -> Optional[Dict[str, float]]:
        """
        テキスト入力に対する簡易感情分析（後方互換性用）
        古いテストコードが辞書形式 {'valence': score} を期待しているためそれに合わせる。
        """
        if not text:
            return None
            
        score = 0.0
        found = False
        for word, val in self.sentiment_lexicon.items():
            if word in text:
                score += val
                found = True
        
        # [修正] floatではなくdictを返す
        return {'valence': score} if found else None

    def update_value_function(self, sensory_input: torch.Tensor, real_reward: float):
        predicted_value = self.value_estimator(sensory_input)
        target = torch.tensor([[real_reward]], device=sensory_input.device)
        return nn.MSELoss()(predicted_value, target)