# ファイルパス: scripts/run_project_health_check.py
# (修正: Check 10 のインラインコードを強化)

import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    logger.info(f"--- 🏃 実行中: {description} ---")
    logger.info(f"コマンド: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info(f"--- ✅ 成功: {description} ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"--- ❌ 失敗: {description} (終了コード: {e.returncode}) ---")
        logger.info(f"  [{description}] エラー出力:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"--- ❌ 予期せぬエラー: {e}")
        return False

def main():
    logger.info("="*60)
    logger.info("🩺 SNNプロジェクト 統合健全性チェック (Runners Path Fixed)")
    logger.info("="*60)
    logger.info(f"📂 作業ディレクトリ: {os.getcwd()}")

    results = {}
    python_cmd = sys.executable

    # Check 1-9 (変更なし) ... 
    # 省略せずに記述する場合は、前回の回答のCheck 1-9と同じコードを使用してください。
    # ここではスペース節約のため省略しますが、実際は全てのチェックを含めます。
    
    # --- 1. Train (Gradient) ---
    results["1. 代理勾配学習 (gradient_based)"] = run_command(
        [python_cmd, "scripts/runners/train.py", "--config", "configs/experiments/smoke_test_config.yaml",
         "--model_config", "configs/models/micro.yaml", "--paradigm", "gradient_based", "--data_path", "data/smoke_test_data.jsonl"],
        "1. 代理勾配学習 (gradient_based)"
    )

    # --- 2. Benchmark ---
    results["2. ベンチマーク実行 (Train+Eval)"] = run_command(
        [python_cmd, "scripts/run_benchmark_suite.py", "--experiment", "health_check_comparison",
         "--epochs", "1", "--batch_size", "2", "--model_config", "configs/models/micro.yaml", "--tag", "HealthCheck"],
        "2. ベンチマーク実行 (Train+Eval)"
    )

    # --- 3. Bio-RL ---
    results["3. 生物学的学習 (Bio-RL)"] = run_command(
        [python_cmd, "scripts/runners/run_rl_agent.py", "--episodes", "2", "--output_dir", "runs/health_check_rl"],
        "3. 生物学的学習 (Bio-RL)"
    )

    # --- 4. Brain Simulation ---
    results["4. 認知アーキテクチャ (ArtificialBrain)"] = run_command(
        [python_cmd, "scripts/runners/run_brain_simulation.py", "--prompt", "Health check prompt", "--model_config", "configs/models/micro.yaml"],
        "4. 認知アーキテクチャ (ArtificialBrain)"
    )

    # --- 5. Efficiency Report ---
    results["5. 効率レポート (Sparsity & T)"] = run_command(
        [python_cmd, "scripts/report_sparsity_and_T.py", "--model_config", "configs/models/micro.yaml", "--data_path", "data/smoke_test_data.jsonl"],
        "5. 効率レポート (Sparsity & T)"
    )

    # --- 6. BitNet ---
    results["6. 1.58bit BitNet学習 (BitRWKV)"] = run_command(
        [python_cmd, "scripts/runners/train.py", "--config", "configs/experiments/smoke_test_config.yaml",
         "--model_config", "configs/models/bit_rwkv_micro.yaml", "--paradigm", "gradient_based", "--data_path", "data/smoke_test_data.jsonl",
         "--override_config", "training.epochs=1"],
        "6. 1.58bit BitNet学習 (BitRWKV)"
    )

    run_command([python_cmd, "scripts/register_demo_experts.py"], "  (MoE Setup: エキスパート登録)")

    results["7. FrankenMoE 構成ビルド"] = run_command(
        [python_cmd, "scripts/manage_models.py", "build-moe", "--keywords", "science,history", "--output", "health_check_moe.yaml"],
        "7. FrankenMoE 構成ビルド"
    )

    results["8. ANN-SNN 変換 (Dry Run)"] = run_command(
        [python_cmd, "scripts/convert_model.py", "--ann_model_path", "runs/dummy_ann.pth", 
         "--snn_model_config", "configs/models/micro.yaml", "--output_snn_path", "runs/converted_snn.pth",
         "--method", "cnn-convert", "--dry-run"],
        "8. ANN-SNN 変換 (Dry Run)"
    )

    results["9. GraphRAG 知識追加"] = run_command(
        [python_cmd, "snn-cli.py", "knowledge", "add", "HealthCheck", "System is healthy", "--vector-store-path", "runs/health_check_vec_store"],
        "9. GraphRAG 知識追加"
    )

    # --- 10. Hardware Compiler ---
    hw_compile_code = """
from snn_research.core.snn_core import SNNCore
from snn_research.hardware.compiler import NeuromorphicCompiler
from omegaconf import OmegaConf
import logging
logging.basicConfig(level=logging.CRITICAL)
cfg = OmegaConf.load('configs/models/micro.yaml')
# 修正: model キーの有無を確認してパラメータを抽出
if 'model' in cfg:
    params = OmegaConf.to_container(cfg.model, resolve=True)
else:
    params = OmegaConf.to_container(cfg, resolve=True)
model = SNNCore(params, vocab_size=100)
compiler = NeuromorphicCompiler()
compiler.compile(model, 'runs/health_check_hardware.yaml')
"""
    results["10. ハードウェアコンパイル"] = run_command(
        [python_cmd, "-c", hw_compile_code],
        "10. ハードウェアコンパイル"
    )

    # --- 11-19 (変更なし) ---
    results["11. Phase 3 統合検証 (SFormer/SEMM)"] = run_command([python_cmd, "scripts/verify_phase3.py"], "11. Phase 3 統合検証 (SFormer/SEMM)")
    results["12. 自律エージェント (Task Solver)"] = run_command([python_cmd, "scripts/runners/run_agent.py", "--task_description", "health_check_task", "--force_retrain", "--model_config", "configs/models/micro.yaml", "--unlabeled_data_path", "data/smoke_test_data.jsonl"], "12. 自律エージェント (Task Solver)")
    results["13. 階層的プランナー (Planner)"] = run_command([python_cmd, "scripts/runners/run_planner.py", "--task_request", "Plan a health check sequence", "--context_data", "System status is unknown"], "13. 階層的プランナー (Planner)")
    results["14. マルチモーダル処理デモ"] = run_command([python_cmd, "scripts/run_multimodal_demo.py"], "14. マルチモーダル処理デモ")
    results["15. 空間認識デモ"] = run_command([python_cmd, "scripts/run_spatial_demo.py"], "15. 空間認識デモ")
    results["16. ECG異常検知デモ"] = run_command([python_cmd, "scripts/run_ecg_analysis.py", "--model_config", "configs/models/ecg_temporal_snn.yaml", "--num_samples", "2", "--time_steps", "100"], "16. ECG異常検知デモ")
    results["17. ハイパーパラメータ最適化 (HPO)"] = run_command([python_cmd, "scripts/runners/run_hpo.py", "--model_config", "configs/models/micro.yaml", "--task", "health_check", "--n_trials", "1", "--eval_epochs", "1", "--metric_name", "accuracy", "--output_base_dir", "runs/health_check_hpo"], "17. ハイパーパラメータ最適化 (HPO)")
    results["18. スパイク活動可視化 (Patterns)"] = run_command([python_cmd, "scripts/visualize_spike_patterns.py", "--model_config", "configs/models/micro.yaml", "--timesteps", "8", "--batch_size", "2", "--output_prefix", "runs/health_check_viz"], "18. スパイク活動可視化 (Patterns)")
    results["19. ニューロンダイナミクス可視化"] = run_command([python_cmd, "scripts/visualize_neuron_dynamics.py", "--model_config", "configs/models/micro.yaml", "--timesteps", "8", "--output_path", "runs/health_check_dynamics.png"], "19. ニューロンダイナミクス可視化")

    logger.info("="*60)
    logger.info("🩺 統合健全性チェック完了")
    logger.info("="*60)
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"総合結果: {passed_count} / {total_count} の主要機能が正常に動作しました。")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status} : {name}")

    if passed_count != total_count:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
