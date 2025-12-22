# ファイルパス: scripts/run_project_health_check.py
# 日本語タイトル: SNNプロジェクト 高精度健全性チェック v3.2 (v20.1 Convergence対応版)
# 目的: プロジェクト全体の各モジュールを実行し、特に v20.1 で導入された蒸留・能動学習ループを検証する。

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.utils import setup_logging, get_device

# ロガーの初期設定
logger = setup_logging(log_dir="logs", log_name="health_check.log")

def run_command(command: str, description: str, cwd: str = ".") -> Tuple[bool, float, str, str]:
    """
    シェルコマンドを実行し、結果と所要時間を返します。
    ModuleNotFoundError 回避のため PYTHONPATH を厳密に制御します。
    """
    logger.info(f"--- 🏃 実行中: {description} ---")
    
    start_time = time.time()
    
    # 環境変数の設定 (PYTHONPATHを絶対パスで固定)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        duration = time.time() - start_time
        stdout = result.stdout
        stderr = result.stderr

        is_success = (result.returncode == 0)
        
        # 成功判定のログ
        if is_success:
            logger.info(f"--- ✅ 成功: {description} ({duration:.2f}s) ---")
        else:
            logger.error(f"--- ❌ 失敗: {description} (Exit Code: {result.returncode}) ---")
            if stderr: logger.error(f"詳細エラー:\n{stderr}")

        return is_success, duration, stdout, stderr

    except Exception as e:
        logger.error(f"実行例外: {e}")
        return False, 0.0, "", str(e)

def verify_logic_artifacts(check_name: str, expected_files: List[str]) -> bool:
    """成果物の生成確認。"""
    missing = [f for f in expected_files if not Path(f).exists()]
    if missing:
        logger.warning(f"  [Logic Check] ⚠️ {check_name}: ファイル欠損: {missing}")
        return False
    logger.info(f"  [Check] ファイル確認OK: {expected_files}")
    return True

def main():
    logger.info("============================================================")
    logger.info("🩺 SNNプロジェクト 健全性チェック v3.2 (v20.1 Convergence)")
    logger.info("============================================================")
    
    device = get_device()
    python_cmd = sys.executable

    # v20.1 の現状に合わせたチェックリスト
    checks = [
        # --- Core Infrastructure ---
        {
            "name": "1. SNNCore v20.5 統計・発火率管理",
            "cmd": f"{python_cmd} -c \"from snn_research.core.snn_core import SNNCore; import torch; cfg={{'architecture_type':'sformer'}}; model=SNNCore(cfg, vocab_size=100); out=model(torch.randint(0,100,(1,16))); print('Firing Rates:', model.get_firing_rates())\"",
            "verify": []
        },
        # --- Cognitive Architecture v20.1 ---
        {
            "name": "2. ArtificialBrain v20.1 (不確実性推定精度)",
            "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain; import torch; brain=ArtificialBrain(); res=brain.run_cognitive_cycle(torch.randn(1,3,224,224)); print('Uncertainty:', res['uncertainty'])\"",
            "verify": []
        },
        {
            "name": "3. 統合蒸留サイクル検証 (v20.1 Core Loop)",
            "cmd": f"{python_cmd} scripts/runners/test_distillation_cycle.py",
            "verify": []
        },
        {
            "name": "4. 能動学習プロトタイプ (Active Inference)",
            "cmd": f"{python_cmd} scripts/runners/run_active_learning_loop.py",
            "verify": []
        },
        # --- Sleep & Memory ---
        {
            "name": "5. SleepConsolidator v2.2 (型安全・蒸留)",
            "cmd": f"{python_cmd} scripts/runners/run_sleep_cycle_demo.py",
            "verify": []
        },
        # --- Models & Training ---
        {
            "name": "6. BitSpikeMamba 1.58bit 学習 (v20.1)",
            # --epochs 1 --max_steps 5 などの引数を追加して、ヘルスチェック用に極小化する
            "cmd": f"{python_cmd} scripts/trainers/train_overfit_demo.py --epochs 1 --max_steps 5",
            "verify": []
        },
        # --- Legacy & Hardware (Stability Check) ---
        {
            "name": "7. 産業用アイ (Industrial Eye v17.0)",
            "cmd": f"{python_cmd} scripts/runners/run_industrial_eye_demo.py",
            "verify": []
        },
        {
            "name": "8. ハードウェアコンパイル (Runs Check)",
            "cmd": f"{python_cmd} -c \"from snn_research.hardware.compiler import NeuromorphicCompiler; print('Compiler version:', NeuromorphicCompiler())\"",
            "verify": []
        }
    ]

    passed_count = 0
    results = []

    for check in checks:
        success, duration, stdout, _ = run_command(check["cmd"], check["name"])
        
        artifact_success = True
        if success and check.get("verify"):
            artifact_success = verify_logic_artifacts(check["name"], check["verify"])

        if success and artifact_success:
            passed_count += 1
            results.append(f"✅ PASS : {check['name']}")
        else:
            results.append(f"❌ FAIL : {check['name']}")
        
        time.sleep(0.2)

    logger.info("============================================================")
    logger.info(f"📊 総合結果: {passed_count} / {len(checks)} 通過")
    for res in results: logger.info(f"  {res}")

    if passed_count == len(checks):
        logger.info("🎉 プロジェクトは v20.1 仕様で健全です。")
        sys.exit(0)
    else:
        logger.error("⚠️ 一部の機能が v20.1 仕様に適合していません。")
        sys.exit(1)

if __name__ == "__main__":
    main()
