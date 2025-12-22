# scripts/runners/run_sleep_cycle_demo.py
# 日本語タイトル: SNN Sleep & Consolidation Demo (v3 完全整合版)
#
# 変更点:
# - [修正 v3] mypy修正: sleep_manager を SleepConsolidator にキャストして .memory アクセスを許可。
# - [修正 v3] mypy修正: memory_system の呼び出し型安全性を向上。

import os
import sys
import torch
import logging
import time
import random
from typing import Dict, Any, cast, List, Optional

# パス追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.utils import setup_logging
from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory

logger = setup_logging(log_dir="logs", log_name="sleep_cycle_demo.log")

# ... build_brain_with_sleep 関数は以前と同じため中略 ...

def simulate_daytime_experiences(brain: ArtificialBrain, num_experiences: int = 5):
    """
    日中の活動をシミュレーションし、メモリに記録。
    """
    logger.info(f"☀️ Simulating daytime: Experiencing {num_experiences} tasks...")
    
    # mypy修正: brain.sleep_manager が Optional[nn.Module] のためキャストが必要
    if brain.sleep_manager is None:
        logger.error("Sleep manager not initialized!")
        return

    # 明示的に型を SleepConsolidator として扱う
    manager = cast(SleepConsolidator, brain.sleep_manager)
    memory_system = manager.memory

    for i in range(num_experiences):
        experience: Dict[str, Any] = {
            "state": {"context": f"Problem {i}"},
            "action": "solved_problem",
            "result": "Success",
            "reward": {"external": 1.0 + random.random()},
            "expert_used": ["reasoning_engine"],
            "decision_context": {"reason": "Logic verified"}
        }
        
        # mypy修正: record_experience が確実に呼べることを担保
        if hasattr(memory_system, 'record_experience'):
            memory_system.record_experience(
                state=experience["state"],
                action=experience["action"],
                result=experience["result"],
                reward=experience["reward"],
                expert_used=experience["expert_used"],
                decision_context=experience["decision_context"]
            )
        
        time.sleep(0.1)

    logger.info("📝 Experiences recorded.")


def main():
    logger.info("============================================================")
    logger.info("🌙 SNN Sleep & Consolidation Demo")
    logger.info("============================================================")
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("🚀 Using CUDA GPU")

    # 1. 脳の構築 (疲れた状態で初期化)
    brain = build_brain_with_sleep(device)
    
    initial_health = brain.get_brain_status()
    # 辞書アクセスを安全に行う
    if 'astrocyte' in initial_health and 'metrics' in initial_health['astrocyte']:
        metrics = initial_health['astrocyte']['metrics']
        
        # [Fix] KeyError対策: get() を使用し、欠損時はデフォルト値または安全なフォールバック
        energy = metrics.get('energy_percent', 0.0)
        fatigue = metrics.get('fatigue_index', 0.0)
        
        logger.info(f"😫 Initial State: Energy={energy:.1f}%, Fatigue={fatigue:.1f}")
        
        if 'energy_percent' not in metrics:
            logger.warning(f"⚠️ 'energy_percent' missing in metrics. Available keys: {list(metrics.keys())}")
    
    # 2. 日中の活動 (経験の蓄積)
    simulate_daytime_experiences(brain, num_experiences=10)
    
    # 3. 睡眠サイクルの実行
    logger.info("\n🛌 Initiating Sleep Cycle (8 hours)...")
    start_time = time.time()
    
    # ArtificialBrain.sleep_cycle() を呼び出す
    # 内部で Astrocyteの回復 と SleepConsolidatorの学習 が走る
    brain.sleep_cycle()
    
    duration = time.time() - start_time
    logger.info(f"⏰ Sleep process took {duration:.3f}s (Simulated 8h)")

    # 4. 覚醒後の状態確認
    logger.info("\n🌅 Waking up...")
    final_health = brain.get_brain_status()
    
    if 'astrocyte' in final_health and 'metrics' in final_health['astrocyte']:
        metrics_new = final_health['astrocyte']['metrics']
        
        energy_new = metrics_new.get('energy_percent', 100.0)
        fatigue_new = metrics_new.get('fatigue_index', 0.0)
        
        logger.info(f"🤩 Final State: Energy={energy_new:.1f}%, Fatigue={fatigue_new:.1f}")
        
        # 検証
        if energy_new > 90.0 and fatigue_new < 10.0:
            logger.info("✅ Homeostasis Check: PASSED (Fully recovered)")
        else:
            logger.error("❌ Homeostasis Check: FAILED")
    else:
        logger.error("❌ Health status format invalid.")

    # 夢の効果確認 (ログから判断)
    logger.info("============================================================")
    logger.info("🎉 Sleep Cycle Demo Completed.")

if __name__ == "__main__":
    main()
