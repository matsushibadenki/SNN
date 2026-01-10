# ファイルパス: snn_research/adaptive/intrinsic_motivator.py
# 日本語タイトル: Intrinsic Motivator (Curiosity & Empowerment)
# 目的・内容:
#   ROADMAP Phase 2.1 "Intrinsic Reward" 対応。
#   外部報酬がない環境で、エージェントの行動原理となる「内発的報酬」を生成する。
#   1. Curiosity (Prediction Error): 予測できないことを知りたい欲求。
#   2. Empowerment (Control Authority): 環境を思い通りに動かしたい欲求。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class IntrinsicMotivator(nn.Module):
    """
    内発的報酬（Curiosity, Empowerment）を計算するモジュール。
    SNNの可塑性制御シグナル(Dopamine equivalent)として機能する。
    """
    
    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__()
        self.curiosity_weight = config.get("curiosity_weight", 1.0)
        self.empowerment_weight = config.get("empowerment_weight", 0.5)
        self.decay_rate = config.get("decay_rate", 0.99) # 驚きに対する慣れ
        
        # 移動平均によるベースライン誤差（慣れのため）
        self.register_buffer("running_error_mean", torch.tensor(0.1))
        
        logger.info("🧠 Intrinsic Motivator initialized.")

    def compute_reward(
        self, 
        predicted_state: torch.Tensor, 
        actual_state: torch.Tensor,
        action_impact: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        内発的報酬を計算する。
        
        Args:
            predicted_state: 世界モデルが予測した次の状態 (z_pred)
            actual_state: 実際に観測された次の状態 (z_actual)
            action_impact: (Optional) 行動による状態変化量 (Empowerment用)
            
        Returns:
            intrinsic_reward: スカラー報酬値
        """
        # 1. Curiosity: Prediction Error (MSE)
        # 予測と現実のズレが大きいほど「驚き」＝「報酬」とする（新しい知識の獲得）
        # ただし、ノイズへの過適合を防ぐため、あまりにランダムなものは除外する工夫が必要だが、
        # ここではシンプルに予測誤差を用いる。
        
        with torch.no_grad():
            prediction_error = F.mse_loss(predicted_state, actual_state, reduction='none').mean(dim=-1)
            # [Batch, Time] -> [Batch] (平均)
            batch_error = prediction_error.mean()
            
            # Update baseline (habituation)
            self.running_error_mean = self.running_error_mean * self.decay_rate + batch_error * (1 - self.decay_rate)
            
            # Normalize curiosity: 普段よりどれだけ驚いたか
            # running_meanより大きい時だけプラスにする（退屈を防ぐ）
            curiosity = torch.relu(batch_error - self.running_error_mean) * 10.0
        
        # 2. Empowerment: Action Impact
        # 自分の行動が環境に変化を与えたか？ (z_t+1 - z_t の大きさなど)
        empowerment = 0.0
        if action_impact is not None:
             empowerment = action_impact.norm(p=2, dim=-1).mean()
             
        total_reward = (self.curiosity_weight * curiosity) + (self.empowerment_weight * empowerment)
        
        return total_reward

    def get_stats(self) -> Dict[str, float]:
        return {
            "baseline_error": self.running_error_mean.item()
        }