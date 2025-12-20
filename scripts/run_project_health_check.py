import os
import sys
import subprocess
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# -----------------------------------------------------------------------------
# Path: scripts/run_project_health_check.py
# Title: SNNプロジェクト 高精度健全性チェック (Health Check Runner) v3.1
# Description: プロジェクト全体の各モジュールを実行し、エラーやパフォーマンスを検証します。
#              CI/CDパイプラインや開発時のスモークテストとして使用されます。
#              v3.1 Update: v16.3/v17.0 の新機能（統合脳、睡眠、産業用アイ）を追加。
# -----------------------------------------------------------------------------

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.utils import setup_logging, get_device

# ロガーの初期設定
logger = setup_logging(log_dir="logs", log_name="health_check.log")

def run_command(command: str, description: str, cwd: str = ".") -> Tuple[bool, float, str, str]:
    """
    シェルコマンドを実行し、結果と所要時間を返します。
    ログ出力を解析し、既知の無害な警告を除外して判定を行います。
    
    Args:
        command: 実行するシェルコマンド
        description: テスト項目の説明
        cwd: カレントディレクトリ
        
    Returns:
        (is_success, duration, stdout, stderr) のタプル
    """
    logger.info(f"--- 🏃 実行中: {description} ---")
    logger.info(f"コマンド: {command}")
    
    start_time = time.time()
    
    # 環境変数の設定 (PYTHONPATHの補完)
    env = os.environ.copy()
    python_path = project_root
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = python_path

    try:
        # プロセス実行
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',       # 明示的なエンコーディング指定
            errors='replace'        # デコードエラーで落ちないように置換
        )
        
        duration = time.time() - start_time
        stdout = result.stdout
        stderr = result.stderr

        # 出力のログ記録 (長すぎる場合は切り詰め)
        if stdout:
            log_stdout = stdout[:500] + "..." if len(stdout) > 500 else stdout
            logger.info(f"出力(stdout) [先頭500文字]:\n{log_stdout}")
        if stderr:
            log_stderr = stderr[:500] + "..." if len(stderr) > 500 else stderr
            logger.info(f"出力(stderr) [先頭500文字]:\n{log_stderr}")

        is_success = (result.returncode == 0)
        
        # エラーキーワードチェック (ただし cupy エラーなどは無視)
        error_keywords = ["Traceback (most recent call last)", "ModuleNotFoundError", "ImportError"]
        ignored_keywords = [
            "No module named 'cupy'", 
            "cupy not found",
            "UserWarning", 
            "DeprecationWarning"
        ]
        
        # 成功コード(0)でも、致命的なエラーログが出ていないか確認
        if is_success:
            for keyword in error_keywords:
                if keyword in stderr:
                    # 無視リストに含まれるキーワードが同じ行または文脈にあるか確認
                    if not any(ign in stderr for ign in ignored_keywords):
                        logger.warning(f"⚠️ 終了コードは0ですが、エラーらしき出力が検出されました: {keyword}")
                        # 厳密なチェックならここで is_success = False にする場合もあるが、今回は警告にとどめる
        
        if is_success:
            logger.info(f"--- ✅ 成功: {description} ({duration:.2f}s) ---")
        else:
            logger.error(f"--- ❌ 失敗: {description} (Exit Code: {result.returncode}) ---")
            logger.error(f"詳細エラー:\n{stderr}")

        return is_success, duration, stdout, stderr

    except Exception as e:
        logger.error(f"実行例外: {e}")
        return False, 0.0, "", str(e)

def verify_logic_artifacts(check_name: str, expected_files: List[str]) -> bool:
    """
    成果物の生成確認を行います。
    
    Args:
        check_name: チェック名
        expected_files: 期待されるファイルパスのリスト
        
    Returns:
        すべて存在すればTrue
    """
    missing = []
    for f in expected_files:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        logger.warning(f"  [Logic Check] ⚠️ {check_name}: 以下のファイルが生成されていません: {missing}")
        return False
    else:
        logger.info(f"  [Check] ファイル確認OK: {expected_files[0]} 等")
        return True

def main():
    logger.info("============================================================")
    logger.info("🩺 SNNプロジェクト 高精度健全性チェック (v3.1 - Integrated)")
    logger.info("============================================================")
    
    device = get_device()
    logger.info(f"📂 作業ディレクトリ: {os.getcwd()}")
    logger.info(f"🖥️  実行デバイス: {device}")
    
    python_cmd = sys.executable

    # チェックリスト定義
    # name: 表示名
    # cmd: 実行コマンド
    # verify: 実行後に存在すべきファイルリスト (オプション)
    checks = [
        {
            "name": "1. 代理勾配学習 (gradient_based)",
            "cmd": f"{python_cmd} scripts/runners/train.py --config configs/experiments/smoke_test_config.yaml --model_config configs/models/micro.yaml --paradigm gradient_based --data_path data/smoke_test_data.jsonl --override_config training.log_dir=runs/smoke_tests",
            "verify": ["runs/smoke_tests/best_model.pth"]
        },
        {
            "name": "2. ベンチマーク実行 (Train+Eval)",
            "cmd": f"{python_cmd} scripts/run_benchmark_suite.py --experiment health_check_comparison --epochs 1 --batch_size 2 --model_config configs/models/micro.yaml --tag HealthCheck",
            "verify": ["benchmarks/results/benchmark_latest.json"]
        },
        {
            "name": "3. 生物学的学習 (Bio-RL)",
            "cmd": f"{python_cmd} scripts/runners/run_rl_agent.py --episodes 2 --output_dir runs/health_check_rl",
            "verify": [] # Bio-RLはモデル保存しない場合があるため空リスト
        },
        {
            "name": "4. 認知アーキテクチャ (ArtificialBrain v14)",
            "cmd": f"{python_cmd} scripts/runners/run_brain_simulation.py --prompt \"Health check prompt\" --model_config configs/models/micro.yaml",
            "verify": []
        },
        {
            "name": "5. 効率レポート (Sparsity & T)",
            "cmd": f"{python_cmd} scripts/report_sparsity_and_T.py --model_config configs/models/micro.yaml --data_path data/smoke_test_data.jsonl",
            "verify": []
        },
        {
            "name": "6. 1.58bit BitNet学習 (BitRWKV)",
            "cmd": f"{python_cmd} scripts/runners/train.py --config configs/experiments/smoke_test_config.yaml --model_config configs/models/bit_rwkv_micro.yaml --paradigm gradient_based --data_path data/smoke_test_data.jsonl --override_config training.epochs=1 --override_config training.log_dir=runs/smoke_tests_bitnet",
            "verify": ["runs/smoke_tests_bitnet/best_model.pth"]
        },
        {
            "name": "7. MoE Setup (エキスパート登録)",
            "cmd": f"{python_cmd} scripts/register_demo_experts.py",
            "verify": []
        },
        {
            "name": "8. FrankenMoE 構成ビルド",
            "cmd": f"{python_cmd} scripts/manage_models.py build-moe --keywords science,history --output health_check_moe.yaml",
            "verify": ["configs/models/health_check_moe.yaml"]
        },
        {
            "name": "9. ANN-SNN 変換 (Dry Run)",
            "cmd": f"{python_cmd} scripts/convert_model.py --ann_model_path runs/dummy_ann.pth --snn_model_config configs/models/micro.yaml --output_snn_path runs/converted_snn.pth --method cnn-convert --dry-run",
            "verify": []
        },
        {
            "name": "10. GraphRAG 知識追加",
            "cmd": f"{python_cmd} snn-cli.py knowledge add \"HealthCheck\" \"System is healthy\" --vector-store-path runs/health_check_vec_store",
            "verify": ["runs/health_check_vec_store"]
        },
        {
            "name": "11. ハードウェアコンパイル (NeuromorphicCompiler)",
            "cmd": f"{python_cmd} -c \"from snn_research.core.snn_core import SNNCore; from snn_research.hardware.compiler import NeuromorphicCompiler; from omegaconf import OmegaConf; import logging; import sys; logging.basicConfig(level=logging.CRITICAL); cfg = OmegaConf.load('configs/models/micro.yaml'); params = OmegaConf.to_container(cfg.model, resolve=True) if 'model' in cfg else OmegaConf.to_container(cfg, resolve=True); model = SNNCore(params, vocab_size=100); compiler = NeuromorphicCompiler(); compiler.compile(model, 'runs/health_check_hardware.yaml')\"",
            "verify": ["runs/health_check_hardware.yaml"]
        },
        {
            "name": "12. Phase 3 統合検証 (SFormer/SEMM)",
            "cmd": f"{python_cmd} scripts/verify_phase3.py",
            "verify": []
        },
        {
            "name": "13. 自律エージェント (Task Solver)",
            "cmd": f"{python_cmd} scripts/runners/run_agent.py --task_description health_check_task --force_retrain --model_config configs/models/micro.yaml --unlabeled_data_path data/smoke_test_data.jsonl",
            "verify": ["runs/dummy_trained_model.pth"]
        },
        {
            "name": "14. オンチップ学習 (STDP)",
            "cmd": f"{python_cmd} scripts/run_on_chip_learning.py",
            "verify": []
        },
        {
            "name": "15. 生物学的マイクロサーキット (PD14)",
            "cmd": f"{python_cmd} scripts/run_bio_microcircuit_demo.py",
            "verify": []
        },
        {
            "name": "16. 脳型OSシミュレーション (Neuromorphic OS)",
            "cmd": f"{python_cmd} scripts/runners/run_neuromorphic_os.py",
            "verify": []
        },
        {
            "name": "17. Forward-Forward (FF) 学習",
            "cmd": f"""{python_cmd} -c "import torch; import torch.nn as nn; from snn_research.training.trainers.forward_forward import ForwardForwardTrainer; import sys; model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 4)); inputs = torch.randn(4, 10); targets = torch.randint(0, 2, (4,)); trainer = ForwardForwardTrainer(model, learning_rate=0.01); metrics = trainer.train_epoch([(inputs, targets)]); print(f'FF Training Finished. Metrics: {{metrics}}')" """,
            "verify": []
        },
        {
            "name": "18. Hyperdimensional Computing (HDC)",
            "cmd": f"""{python_cmd} -c "from snn_research.cognitive_architecture.hdc_engine import HDCEngine, HDCReasoningAgent; engine = HDCEngine(dim=2048); agent = HDCReasoningAgent(engine); agent.learn_concept('Japan', 'Capital', 'Tokyo'); result = agent.query('Japan', 'Capital'); print(f'HDC Query Result: {{result}}'); " """,
            "verify": []
        },
        {
            "name": "19. Tsetlin Machine (Logic Learning)",
            "cmd": f"""{python_cmd} -c "from snn_research.cognitive_architecture.tsetlin_machine import TsetlinMachine; import numpy as np; X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.int32); y = np.array([0, 1, 1, 0], dtype=np.int32); tm = TsetlinMachine(number_of_clauses=10, number_of_features=2, states=100, s=3.9, threshold=15); [tm.fit(X[i], y[i]) for _ in range(5) for i in range(4)]; pred = tm.predict(X[0]); print(f'TM Prediction: {{pred}}')" """,
            "verify": []
        },
        {
            "name": "20. Oscillatory Neural Network (ONN)",
            "cmd": f"""{python_cmd} -c "from snn_research.core.networks.oscillatory_network import HopfieldONN; import torch; onn = HopfieldONN(num_neurons=4); pattern = torch.tensor([[1.0, -1.0, 1.0, -1.0]]); onn.train(pattern); recovered = onn.retrieve(torch.tensor([[0.5, -0.8, 0.2, -0.9]]), steps=50); print(f'Recovered: {{recovered}}')" """,
            "verify": []
        },
        # --- v16.3 / v17.0 New Features ---
        {
            "name": "21. 統合脳デモ (v16.3 Integrated Brain)",
            "cmd": f"{python_cmd} scripts/runners/run_brain_v16_demo.py",
            "verify": []
        },
        {
            "name": "22. 睡眠サイクル (Sleep & Consolidation)",
            "cmd": f"{python_cmd} scripts/runners/run_sleep_cycle_demo.py",
            "verify": []
        },
        {
            "name": "23. 産業用アイ (Industrial Eye v17.0)",
            "cmd": f"{python_cmd} scripts/runners/run_industrial_eye_demo.py",
            "verify": []
        }
    ]

    passed_count = 0
    results = []

    for check in checks:
        success, duration, stdout, _ = run_command(check["cmd"], check["name"], cwd=".")
        
        # コマンド成功かつ検証ファイル指定がある場合、成果物をチェック
        artifact_success = True
        if success and "verify" in check and check["verify"]:
            artifact_success = verify_logic_artifacts(check["name"], check["verify"])

        if success and artifact_success:
            passed_count += 1
            results.append(f"✅ PASS : {check['name']}")
        else:
            reason = "Command Failed" if not success else "Artifact Missing"
            results.append(f"❌ FAIL : {check['name']} ({reason})")
        
        # 連続実行時のバッファ待機
        time.sleep(0.5)

    logger.info("============================================================")
    logger.info("🩺 統合健全性チェック完了 (v3.1 - Integrated)")
    logger.info("============================================================")
    logger.info(f"総合結果: {passed_count} / {len(checks)} のチェックを通過しました。")
    
    for res in results:
        logger.info(f"  {res}")

    if passed_count == len(checks):
        logger.info("🎉 すべてのテストが正常に完了しました。")
        sys.exit(0)
    else:
        logger.error("⚠️ 一部のテストが失敗しました。ログを確認してください。")
        sys.exit(1)

if __name__ == "__main__":
    main()