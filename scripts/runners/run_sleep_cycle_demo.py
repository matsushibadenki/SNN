# ファイルパス: scripts/runners/run_sleep_cycle_demo.py
# 日本語タイトル: SNN Sleep & Consolidation Demo (完全版)
# 目的: 睡眠と記憶固定化プロセスのデモ実行。NameError の解消。

import os
import sys
import torch
import logging
import time
from typing import Dict, Any, cast

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.utils import setup_logging
logger = setup_logging(log_dir="logs", log_name="sleep_cycle_demo.log")

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory

def build_brain_with_sleep(device: str = 'cpu') -> ArtificialBrain:
    """[修正] スコープ外であった関数を明示的に定義"""
    logger.info("🧠 Initializing Artificial Brain with Sleep Capabilities...")
    
    # 依存コンポーネントの構築
    workspace = GlobalWorkspace()
    memory_system = Memory(rag_system=None, memory_path="runs/demo_sleep_memory.jsonl")
    
    # SleepConsolidatorの初期化
    sleep_manager = SleepConsolidator(
        memory_system=memory_system,
        device=device
    )
    
    brain = ArtificialBrain(device=device)
    # 動的に sleep_manager を追加
    setattr(brain, 'sleep_manager', sleep_manager)
    
    return brain

def simulate_daytime_experiences(brain: ArtificialBrain, num_experiences: int = 5):
    """日中の経験シミュレーション"""
    logger.info(f"☀️ Simulating {num_experiences} experiences...")
    
    manager = getattr(brain, 'sleep_manager', None)
    if manager is None: return
    
    memory_system = cast(SleepConsolidator, manager).memory

    for i in range(num_experiences):
        if hasattr(memory_system, 'record_experience'):
            memory_system.record_experience(
                state={"context": f"Task {i}"},
                action="process",
                result="Success",
                reward={"external": 1.0},
                expert_used=["core"],
                decision_context={"reason": "demo"}
            )
        time.sleep(0.01)

def main():
    logger.info("============================================================")
    logger.info("🌙 SNN Sleep & Consolidation Demo")
    logger.info("============================================================")
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # build_brain_with_sleep が定義されたため呼び出し可能
    brain = build_brain_with_sleep(device)
    
    # 経験の記録
    simulate_daytime_experiences(brain, num_experiences=5)
    
    # 睡眠サイクルの実行
    logger.info("\n🛌 Initiating Sleep Cycle...")
    manager = getattr(brain, 'sleep_manager', None)
    if manager:
        cast(SleepConsolidator, manager).perform_sleep_cycle()

    logger.info("============================================================")
    logger.info("🎉 Sleep Cycle Demo Completed.")

if __name__ == "__main__":
    main()
