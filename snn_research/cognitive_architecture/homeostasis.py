# ファイルパス: snn_research/cognitive_architecture/homeostasis.py
# 日本語タイトル: Homeostasis System (Biological Constraints)
# 目的・内容:
#   ROADMAP Phase 2 "Autonomy" の基盤。
#   エージェントの内部状態（エネルギー、疲労、ストレス）を管理し、
#   「いつ寝るべきか」「いつ探索すべきか」といった行動調整シグナルを生成する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Homeostasis(nn.Module):
    """
    自律エージェントの恒常性維持システム。
    疲労(Fatigue)が溜まると睡眠欲求(Sleep Pressure)が高まる。
    """
    
    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__()
        
        # Parameters
        self.max_energy = config.get("max_energy", 100.0)
        self.fatigue_rate = config.get("fatigue_rate", 0.5) # 1ステップあたりの疲労蓄積
        self.recovery_rate = config.get("recovery_rate", 5.0) # 睡眠中の回復速度
        self.sleep_threshold = config.get("sleep_threshold", 80.0) # 眠くなる疲労度
        
        # State (Buffers to be saved with model)
        self.register_buffer("energy", torch.tensor(self.max_energy))
        self.register_buffer("fatigue", torch.tensor(0.0))
        self.register_buffer("cycle_count", torch.tensor(0)) # 経過日数
        
        logger.info("💓 Homeostasis System initialized.")

    def update(self, action_intensity: float = 1.0) -> Dict[str, float]:
        """
        活動時の状態更新。
        行動が激しいほど疲労が溜まる。
        """
        # 疲労蓄積
        fatigue_increase = self.fatigue_rate * action_intensity
        self.fatigue = torch.clamp(self.fatigue + fatigue_increase, 0, 100)
        
        # エネルギー消費
        self.energy = torch.clamp(self.energy - (fatigue_increase * 0.5), 0, self.max_energy)
        
        return self.get_status()

    def rest(self) -> Dict[str, float]:
        """
        休息（睡眠）時の状態更新。
        疲労が回復する。
        """
        self.fatigue = torch.clamp(self.fatigue - self.recovery_rate, 0, 100)
        self.energy = torch.clamp(self.energy + (self.recovery_rate * 0.2), 0, self.max_energy)
        
        return self.get_status()

    def check_needs(self) -> str:
        """
        現在の最も優先すべき欲求を返す。
        """
        if self.fatigue > self.sleep_threshold:
            return "sleep"
        elif self.energy < 20.0:
            return "recharge"
        else:
            return "explore"

    def new_day(self):
        """新しい一日を開始（カウンタ更新）"""
        self.cycle_count += 1
        logger.info(f"🌅 Day {self.cycle_count.item()} started. Fatigue reset.")

    def get_status(self) -> Dict[str, float]:
        return {
            "energy": self.energy.item(),
            "fatigue": self.fatigue.item(),
            "cycle": self.cycle_count.item()
        }