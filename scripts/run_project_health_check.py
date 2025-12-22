# ファイルパス: scripts/run_project_health_check.py
# 日本語タイトル: SNNプロジェクト 全機能網羅ヘルスチェック v4.1 (エラー修正版)

import os
import sys
import subprocess
import time
import logging
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> Tuple[bool, float]:
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    try:
        result = subprocess.run(command, shell=True, env=env, capture_output=True, text=True, timeout=60)
        duration = time.time() - start_time
        if result.returncode == 0:
            logger.info(f"✅ 成功: {description} ({duration:.2f}s)")
            return True, duration
        else:
            logger.error(f"❌ 失敗: {description}\n[STDERR]: {result.stderr}")
            return False, duration
    except Exception as e:
        logger.error(f"⚠️ 例外: {description} ({e})")
        return False, 0.0

def main():
    logger.info("🩺 SNNプロジェクト ヘルスチェック v4.1 開始")
    python_cmd = sys.executable

    # 修正されたチェックリスト
    checks = [
        # 1. Core: 引数を辞書やデフォルト値で補完
        {
            "name": "Core: SNNCore & SFormer",
            "cmd": f"{python_cmd} -c \"from snn_research.core.snn_core import SNNCore; import torch; SNNCore({{'architecture_type':'sformer'}}, 100)(torch.randint(0,100,(1,16)))\""
        },
        {
            "name": "Core: BitSpikeMamba 1.58bit",
            "cmd": f"{python_cmd} -c \"from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba; import torch; m=BitSpikeMamba(vocab_size=100, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=1, time_steps=4, neuron_config={{'type':'lif'}}); m(torch.randint(0,100,(1,4)))\""
        },
        # 2. Cognitive: run_cognitive_cycle に入力を追加
        {
            "name": "Cognitive: ArtificialBrain Cycle",
            "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain; import torch; brain=ArtificialBrain(); brain.run_cognitive_cycle(torch.randn(1,3,32,32))\""
        },
        {
            "name": "Cognitive: Sleep & Consolidation",
            "cmd": f"{python_cmd} scripts/runners/run_sleep_cycle_demo.py"
        },
        {
            "name": "Cognitive: Reasoning & Meta-cognition",
            "cmd": f"{python_cmd} scripts/runners/run_reasoning_to_sleep_demo.py"
        },
        # 3. Agent: PlannerSNN の引数を修正
        {
            "name": "Agent: Planner SNN",
            "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.planner_snn import PlannerSNN; import torch; p=PlannerSNN(vocab_size=100, d_model=64, d_state=16, num_layers=1, time_steps=4, n_head=2, num_skills=5, neuron_config={{'type':'lif'}}); p(torch.randint(0,100,(1,4)))\""
        },
        # 他、App, Train, HPO, Hardware 等は成功していたため維持
        {"name": "App: Industrial Eye (DVS)", "cmd": f"{python_cmd} scripts/runners/run_industrial_eye_demo.py"},
        {"name": "App: ECG Temporal Analysis", "cmd": f"{python_cmd} scripts/run_ecg_analysis.py"},
        {"name": "Train: Overfit Smoke Test", "cmd": f"{python_cmd} scripts/trainers/train_overfit_demo.py --epochs 1 --max_steps 2"},
        {"name": "Hardware: Neuromorphic Compiler", "cmd": f"{python_cmd} -c \"from snn_research.hardware.compiler import NeuromorphicCompiler; NeuromorphicCompiler()\""},
    ]

    passed = 0
    for check in checks:
        success, _ = run_command(check["cmd"], check["name"])
        if success: passed += 1
    
    logger.info(f"📊 最終結果: {passed} / {len(checks)} 項目合格")
    sys.exit(0 if passed == len(checks) else 1)

if __name__ == "__main__":
    main()
