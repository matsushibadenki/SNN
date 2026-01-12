# ファイルパス: scripts/experiments/brain/run_phase2_autonomous_agent.py
# Title: Phase 2 Autonomous Agent Experiment
# 修正内容: Mypyエラー修正 (型ヒントの追加)。

import torch
import logging
import sys
import os
import time
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from snn_research.core.snn_core import SNNCore
from snn_research.adaptive.active_inference_agent import ActiveInferenceAgent
from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase2AutonomousAgent:
    """Phase 2: 自律学習エージェント"""
    def __init__(self):
        self.device = "cpu"
        self.input_dim = 64
        self.hidden_dim = 128
        self.output_dim = 10
        
        # コア脳 (SNN)
        self.brain = SNNCore(
            in_features=self.input_dim,
            hidden_features=self.hidden_dim,
            out_features=self.output_dim
        ).to(self.device)
        
        # 能動的推論モジュール
        self.active_inference = ActiveInferenceAgent(
            state_dim=self.hidden_dim,
            action_dim=self.output_dim
        )
        
        # 内発的動機づけ
        self.motivator = IntrinsicMotivator()
        
        # 知識ベース
        # [Mypy Fix] 型ヒントを追加
        self.knowledge_base: List[Dict[str, Any]] = [] 
        
        logger.info("🧠 Autonomous Agent initialized.")

    def run_life_cycle(self, steps: int = 100):
        """ライフサイクルの実行"""
        logger.info(f"Starting life cycle for {steps} steps...")
        
        for t in range(steps):
            # 1. 環境からの入力 (ダミー)
            sensory_input = torch.randn(1, self.input_dim).to(self.device)
            
            # 2. 脳による処理 (知覚)
            brain_state = self.brain(sensory_input)
            
            # 3. 能動的推論 (行動選択)
            action = self.active_inference.select_action(brain_state)
            
            # 4. 内発的報酬の計算 (好奇心)
            intrinsic_reward = self.motivator.calculate_reward(brain_state)
            
            # 5. 学習 (可塑性更新)
            if intrinsic_reward > 0.5:
                # 驚きが大きい場合、学習を強化
                target = torch.randn(1, self.output_dim).to(self.device) # ダミー
                self.brain.update_plasticity(sensory_input, target, learning_rate=0.05)
                
                # 知識の蓄積
                self.knowledge_base.append({
                    "step": t,
                    "input_summary": sensory_input.mean().item(),
                    "surprise": intrinsic_reward
                })
            
            if (t+1) % 20 == 0:
                logger.info(f"Step {t+1}: Intrinsic Reward={intrinsic_reward:.4f}, Knowledge={len(self.knowledge_base)}")
                
        logger.info("✅ Life cycle completed successfully.")

if __name__ == "__main__":
    agent = Phase2AutonomousAgent()
    agent.run_life_cycle()