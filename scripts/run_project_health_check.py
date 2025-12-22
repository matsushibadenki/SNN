# ファイルパス: scripts/run_project_health_check.py
# 日本語タイトル: SNNプロジェクト 全機能網羅ヘルスチェック v4.0 (v20.5対応版)
# 目的: プロジェクトに含まれる全コンポーネント（Core, Cognitive, Models, scripts）の健全性を検証する。

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> Tuple[bool, float]:
    """コマンドを実行し、成功か否かと所要時間を返す。"""
    logger.info(f"--- 🏃 実行中: {description} ---")
    start_time = time.time()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    try:
        # ヘルスチェックのため、長時間実行されないようタイムアウトを設定（必要に応じて調整）
        result = subprocess.run(
            command, shell=True, env=env, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace'
        )
        duration = time.time() - start_time
        if result.returncode == 0:
            logger.info(f"✅ 成功: {description} ({duration:.2f}s)")
            return True, duration
        else:
            logger.error(f"❌ 失敗: {description}\n[STDOUT]: {result.stdout}\n[STDERR]: {result.stderr}")
            return False, duration
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ タイムアウト: {description}")
        return False, 60.0
    except Exception as e:
        logger.error(f"⚠️ 例外発生: {description} ({e})")
        return False, 0.0

def main():
    logger.info("============================================================")
    logger.info("🩺 SNNプロジェクト 全機能網羅ヘルスチェック v4.0")
    logger.info("============================================================")
    
    python_cmd = sys.executable

    # 実装されている全機能を網羅するチェックリスト
    checks = [
        # 1. コア・インフラ
        {"name": "Core: SNNCore & SFormer (v20.5)", "cmd": f"{python_cmd} -c \"from snn_research.core.snn_core import SNNCore; import torch; model=SNNCore({{'architecture_type':'sformer'}}, vocab_size=100); model(torch.randint(0,100,(1,16)))\""},
        {"name": "Core: BitSpikeMamba 1.58bit", "cmd": f"{python_cmd} -c \"from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba; import torch; m=BitSpikeMamba(128, 64); m(torch.randn(1, 10, 128))\""},
        
        # 2. 認知アーキテクチャ (v20.1+)
        {"name": "Cognitive: ArtificialBrain Cycle", "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain; brain=ArtificialBrain(); brain.run_cognitive_cycle()\""},
        {"name": "Cognitive: Sleep & Consolidation", "cmd": f"{python_cmd} scripts/runners/run_sleep_cycle_demo.py"},
        {"name": "Cognitive: Knowledge Distillation", "cmd": f"{python_cmd} scripts/runners/test_distillation_cycle.py"},
        {"name": "Cognitive: Active Inference Loop", "cmd": f"{python_cmd} scripts/runners/run_active_learning_loop.py"},
        {"name": "Cognitive: Reasoning & Meta-cognition", "cmd": f"{python_cmd} scripts/runners/run_reasoning_to_sleep_demo.py"},

        # 3. 各種エージェント
        {"name": "Agent: Digital Life Form", "cmd": f"{python_cmd} scripts/runners/run_life_form.py --duration 2"},
        {"name": "Agent: RL (Reinforcement Learning)", "cmd": f"{python_cmd} scripts/runners/run_rl_agent.py --episodes 2"},
        {"name": "Agent: Planner SNN", "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.planner_snn import PlannerSNN; PlannerSNN()\""},

        # 4. 産業用・特化型モデル
        {"name": "App: Industrial Eye (DVS)", "cmd": f"{python_cmd} scripts/runners/run_industrial_eye_demo.py"},
        {"name": "App: ECG Temporal Analysis", "cmd": f"{python_cmd} scripts/run_ecg_analysis.py"},
        {"name": "App: Spatial/Visual Perception", "cmd": f"{python_cmd} scripts/run_spatial_demo.py"},

        # 5. 学習・最適化
        {"name": "Train: Overfit Smoke Test", "cmd": f"{python_cmd} scripts/trainers/train_overfit_demo.py --epochs 1 --max_steps 2"},
        {"name": "HPO: Auto-tune Efficiency", "cmd": f"{python_cmd} scripts/auto_tune_efficiency.py --n-trials 1"},

        # 6. 変換・ハードウェア
        {"name": "Hardware: Neuromorphic Compiler", "cmd": f"{python_cmd} -c \"from snn_research.hardware.compiler import NeuromorphicCompiler; NeuromorphicCompiler()\""},
        {"name": "Hardware: Event Driven Simulator", "cmd": f"{python_cmd} scripts/run_hardware_simulation.py"},
    ]

    results = []
    passed = 0
    for check in checks:
        success, _ = run_command(check["cmd"], check["name"])
        if success:
            passed += 1
            results.append(f"✅ PASS: {check['name']}")
        else:
            results.append(f"❌ FAIL: {check['name']}")
        time.sleep(0.1)

    logger.info("============================================================")
    logger.info(f"📊 最終結果: {passed} / {len(checks)} 項目合格")
    for r in results: logger.info(f"  {r}")

    if passed == len(checks):
        logger.info("🎉 全機能の健全性が確認されました。")
        sys.exit(0)
    else:
        logger.error("⚠️ 一部の機能で不具合が検出されました。")
        sys.exit(1)

if __name__ == "__main__":
    main()
