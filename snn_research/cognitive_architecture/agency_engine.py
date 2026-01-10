# ファイルパス: snn_research/cognitive_architecture/agency_engine.py
# 日本語タイトル: Agency Engine (Free Will / Veto Mechanism) v1.0
# 目的・内容:
#   ROADMAP Phase 4.3 "Free Will & Agency" 対応。
#   ボトムアップな衝動（Impulse）に対して、トップダウンな意志（Intention）が
#   介入し、行動を許可または拒否（Veto）するメカニズム。
#   「自分の行動を選んでいる」という感覚（Sense of Agency）の基礎となる。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AgencyEngine(nn.Module):
    """
    自由意志エンジン。
    無意識の衝動(Impulse)と、意識的な意図(Intention)を照合し、
    最終的な行動実行指令(Motor Command)を発行する。
    """
    
    def __init__(self, action_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        
        # 1. Evaluator (行動の是非を問う)
        # 入力: [Action_Impulse, Long_Term_Goal]
        # 出力: Veto Probability (拒否確率)
        self.evaluator = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 内部状態: Sense of Agency (自分がやった感)
        self.register_buffer("sense_of_agency", torch.tensor(0.5))
        
        logger.info("🕊️ Agency Engine (Free Will / Veto Power) initialized.")

    def forward(
        self, 
        impulse: torch.Tensor, # 無意識からの行動提案 (Action Vector)
        goal_context: torch.Tensor # 長期的な目標/価値観 (Context Vector)
    ) -> Dict[str, Any]:
        """
        行動提案を審査する。
        """
        # 評価入力の作成
        combined = torch.cat([impulse, goal_context], dim=-1)
        
        # 拒否権の発動確率 (Veto Probability)
        # 高いほど「それはダメだ」と判断している
        veto_prob = self.evaluator(combined)
        
        # 決定 (Thresholding)
        # 確率的な揺らぎを持たせることで「迷い」を表現
        decision_threshold = 0.5
        is_vetoed = veto_prob > decision_threshold
        
        # 最終行動
        if is_vetoed:
            final_action = torch.zeros_like(impulse) # 行動抑制
            status = "VETOED"
        else:
            final_action = impulse # 行動許可
            status = "EXECUTED"
            
        # Sense of Agencyの更新
        # 自分の意図(Goal)と実際の行動(Final)が一致していれば「自己効力感」が上がる
        # ここでは簡易的に「Vetoが成功した」または「意図通り動いた」場合に上昇
        # 衝動的に動いてしまった(Veto失敗)場合は下がるロジック等を入れられるが今回は割愛
        
        return {
            "final_action": final_action,
            "veto_prob": veto_prob.item(),
            "status": status,
            "impulse_strength": impulse.norm().item()
        }