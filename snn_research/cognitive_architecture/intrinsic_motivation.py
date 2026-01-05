# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# 日本語タイトル: Intrinsic Motivation System v2.1.1 (Type Fix)
# 目的・内容:
#   ROADMAP v16.3 "Autonomy & Motivation" の実装。
#   mypyエラー修正: get_internal_state内での辞書型アノテーションを修正し、
#   float型のdrives辞書にstr型の値を混在させられるように対応。

import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class IntrinsicMotivationSystem(nn.Module):
    """
    AIの内発的動機（感情・欲求）を生成するエンジン。
    AsyncBrainKernel (v2.x) と ArtificialBrain (v16.x) の両方に対応。
    """
    def __init__(
        self,
        curiosity_weight: float = 1.0,
        boredom_decay: float = 0.995,
        boredom_threshold: float = 0.8,
        homeostasis_weight: float = 2.0
    ):
        super().__init__()
        self.curiosity_weight = curiosity_weight
        self.boredom_decay = boredom_decay
        self.boredom_threshold = boredom_threshold
        
        # 状態履歴（退屈判定用）
        self.last_input_hash: Optional[int] = None
        self.repetition_count = 0
        
        # 現在の動機状態 (0.0 - 1.0)
        self.drives: Dict[str, float] = {
            "curiosity": 0.5,    # 知的好奇心
            "boredom": 0.0,      # 退屈 (new)
            "survival": 0.0,     # 生存本能
            "comfort": 0.0,      # 快適さ
            "competence": 0.3    # 有能感
        }
        
        logger.info("🔥 Intrinsic Motivation System v2.2 (Hybrid Compatible) initialized.")

    def process(self, input_payload: Any) -> Optional[Dict[str, float]]:
        """
        AsyncBrainKernel用のインターフェース。
        入力に基づいて驚き(Surprise)を計算し、動機を更新する。
        """
        # テキスト入力の場合の簡易的なSurprise計算
        if isinstance(input_payload, str):
            input_hash = hash(input_payload)
            
            # 1. Surprise / Repetition Check
            if input_hash == self.last_input_hash:
                # 同じ入力が続いた -> 予測通り -> Surprise低下、Boredom上昇
                self.repetition_count += 1
                surprise = 0.0
                boredom_delta = 0.2 * self.repetition_count
            else:
                # 新しい入力 -> Surprise上昇、Boredomリセット
                self.repetition_count = 0
                surprise = 1.0
                boredom_delta = -0.5
            
            self.last_input_hash = input_hash
            
            # 値の更新
            self.drives["curiosity"] = self.drives["curiosity"] * 0.9 + surprise * 0.1
            self.drives["boredom"] = float(np.clip(self.drives["boredom"] + boredom_delta, 0.0, 1.0))
            
            # ログ出力
            if self.drives["boredom"] > 0.8:
                logger.warning(f"🥱 Boredom Level Critical: {self.drives['boredom']:.2f} (Seeking Novelty)")
            elif surprise > 0.8:
                logger.info("✨ High Surprise Detected! Curiosity Triggered.")
                
            return {
                "surprise": surprise,
                "boredom": self.drives["boredom"],
                "curiosity_drive": self.drives["curiosity"]
            }
            
        return None

    # --- Methods for ArtificialBrain (Legacy/Full Support) ---

    def update_drives(self, surprise: float, energy_level: float, fatigue_level: float, task_success: bool = False) -> Dict[str, float]:
        """環境状態に基づいて動機を更新 (ArtificialBrain互換)"""
        if surprise > 0.1:
            self.drives["curiosity"] = min(1.0, self.drives["curiosity"] + 0.05)
        else:
            self.drives["curiosity"] = max(0.0, self.drives["curiosity"] - 0.2)
            
        self.drives["survival"] = max(0.0, 1.0 - (energy_level / 1000.0))
        return self.drives

    def get_internal_state(self) -> Dict[str, Any]:
        """状態取得"""
        return dict(self.drives)