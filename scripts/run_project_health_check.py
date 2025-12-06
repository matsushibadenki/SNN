# ファイルパス: scripts/run_project_health_check.py
# Title: SNNプロジェクト 統合健全性チェック (高精度版 v2.2)
# Description: 
#   各コンポーネントの実行だけでなく、生成された成果物やログ内容を検証し、
#   システムの健全性をより厳密に診断する。
#   修正: 標準出力(stdout)だけでなく標準エラー出力(stderr)も結合して検証するように変更。

import subprocess
import sys
import logging
import os
from typing import Callable, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: List[str], description: str, validator: Optional[Callable[[str], bool]] = None) -> bool:
    """
    コマンドを実行し、終了コードとバリデータによる検証を行う。
    """
    logger.info(f"--- 🏃 実行中: {description} ---")
    logger.info(f"コマンド: {' '.join(command)}")
    
    try:
        # 実行 (stdoutとstderrの両方をキャプチャ)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        # --- 修正: 検証用に stdout と stderr を結合 ---
        combined_output = result.stdout + "\n" + result.stderr
        # -----------------------------------------
        
        if result.stdout.strip():
            logger.info(f"出力(stdout): {result.stdout[:500]}..." if len(result.stdout) > 500 else f"出力(stdout): {result.stdout}")
            
        if result.stderr.strip():
            logger.info(f"出力(stderr): {result.stderr[:500]}..." if len(result.stderr) > 500 else f"出力(stderr): {result.stderr}")

        # 追加の検証ロジック
        if validator:
            if not validator(combined_output):
                logger.error(f"--- ❌ 検証失敗: {description} (出力内容または成果物が期待と異なります) ---")
                return False

        logger.info(f"--- ✅ 成功: {description} ---")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"--- ❌ 実行失敗: {description} (終了コード: {e.returncode}) ---")
        logger.error(f"エラー出力:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"--- ❌ 予期せぬエラー: {e}")
        return False

# --- バリデータ関数群 ---

def check_file_exists(filepath: str) -> Callable[[str], bool]:
    def _check(_: str) -> bool:
        exists = os.path.exists(filepath)
        if not exists:
            logger.error(f"  [Check] ファイルが見つかりません: {filepath}")
        else:
            logger.info(f"  [Check] ファイル確認OK: {filepath}")
        return exists
    return _check

def check_log_contains(keyword: str, case_sensitive: bool = False) -> Callable[[str], bool]:
    def _check(output: str) -> bool:
        if case_sensitive:
            contains = keyword in output
        else:
            contains = keyword.lower() in output.lower()
            
        if not contains:
            logger.error(f"  [Check] ログにキーワード '{keyword}' が見つかりません。")
        return contains
    return _check

def check_training_success(log_dir: str) -> Callable[[str], bool]:
    def _check(output: str) -> bool:
        if "nan" in output.lower():
            logger.error("  [Check] 損失(Loss)が NaN になっています。")
            return False
        
        best_model = os.path.join(log_dir, "best_model.pth")
        if not os.path.exists(best_model):
            logger.error(f"  [Check] ベストモデルが保存されていません: {best_model}")
            return False
            
        logger.info(f"  [Check] 学習正常終了確認 (Model: {best_model})")
        return True
    return _check

def main():
    logger.info("="*60)
    logger.info("🩺 SNNプロジェクト 高精度健全性チェック (v2.2)")
    logger.info("="*60)
    logger.info(f"📂 作業ディレクトリ: {os.getcwd()}")

    results = {}
    python_cmd = sys.executable
    
    # --- 1. Train ---
    log_dir_1 = "runs/smoke_tests"
    results["1. 代理勾配学習 (gradient_based)"] = run_command(
        [python_cmd, "scripts/runners/train.py", 
         "--config", "configs/experiments/smoke_test_config.yaml",
         "--model_config", "configs/models/micro.yaml", 
         "--paradigm", "gradient_based", 
         "--data_path", "data/smoke_test_data.jsonl",
         "--override_config", f"training.log_dir={log_dir_1}"],
        "1. 代理勾配学習 (gradient_based)",
        validator=check_training_success(log_dir_1)
    )

    # --- 2. Benchmark ---
    results["2. ベンチマーク実行 (Train+Eval)"] = run_command(
        [python_cmd, "scripts/run_benchmark_suite.py", 
         "--experiment", "health_check_comparison",
         "--epochs", "1", 
         "--batch_size", "2", 
         "--model_config", "configs/models/micro.yaml", 
         "--tag", "HealthCheck"],
        "2. ベンチマーク実行 (Train+Eval)",
        validator=check_log_contains("accuracy")
    )

    # --- 3. Bio-RL ---
    rl_out_dir = "runs/health_check_rl"
    results["3. 生物学的学習 (Bio-RL)"] = run_command(
        [python_cmd, "scripts/runners/run_rl_agent.py", 
         "--episodes", "2", 
         "--output_dir", rl_out_dir],
        "3. 生物学的学習 (Bio-RL)",
        validator=lambda x: os.path.isdir(rl_out_dir) and "Final Average Reward" in x
    )

    # --- 4. Brain Simulation ---
    results["4. 認知アーキテクチャ (ArtificialBrain)"] = run_command(
        [python_cmd, "scripts/runners/run_brain_simulation.py", 
         "--prompt", "Health check prompt", 
         "--model_config", "configs/models/micro.yaml"],
        "4. 認知アーキテクチャ (ArtificialBrain)",
        validator=check_log_contains("認知サイクル完了")
    )

    # --- 5. Efficiency Report ---
    results["5. 効率レポート (Sparsity & T)"] = run_command(
        [python_cmd, "scripts/report_sparsity_and_T.py", 
         "--model_config", "configs/models/micro.yaml", 
         "--data_path", "data/smoke_test_data.jsonl"],
        "5. 効率レポート (Sparsity & T)",
        validator=check_log_contains("Total Spikes")
    )

    # --- 6. BitNet ---
    bitnet_log_dir = "runs/smoke_tests_bitnet"
    results["6. 1.58bit BitNet学習 (BitRWKV)"] = run_command(
        [python_cmd, "scripts/runners/train.py", 
         "--config", "configs/experiments/smoke_test_config.yaml",
         "--model_config", "configs/models/bit_rwkv_micro.yaml", 
         "--paradigm", "gradient_based", 
         "--data_path", "data/smoke_test_data.jsonl",
         "--override_config", "training.epochs=1",
         "--override_config", f"training.log_dir={bitnet_log_dir}"],
        "6. 1.58bit BitNet学習 (BitRWKV)",
        validator=check_training_success(bitnet_log_dir)
    )

    # エキスパート登録
    run_command([python_cmd, "scripts/register_demo_experts.py"], "  (MoE Setup: エキスパート登録)")

    # --- 7. FrankenMoE ---
    moe_config_name = "health_check_moe.yaml"
    moe_config_path = os.path.join("configs", "models", moe_config_name)
    if os.path.exists(moe_config_path): os.remove(moe_config_path)

    results["7. FrankenMoE 構成ビルド"] = run_command(
        [python_cmd, "scripts/manage_models.py", "build-moe", 
         "--keywords", "science,history", 
         "--output", moe_config_name],
        "7. FrankenMoE 構成ビルド",
        validator=check_file_exists(moe_config_path)
    )

    # --- 8. ANN-SNN Conversion ---
    conv_snn_path = "runs/converted_snn.pth"
    dummy_ann_path = "runs/dummy_ann.pth"
    if not os.path.exists(dummy_ann_path):
        import torch
        torch.save(torch.nn.Linear(10, 10).state_dict(), dummy_ann_path)

    results["8. ANN-SNN 変換 (Dry Run)"] = run_command(
        [python_cmd, "scripts/convert_model.py", 
         "--ann_model_path", dummy_ann_path, 
         "--snn_model_config", "configs/models/micro.yaml", 
         "--output_snn_path", conv_snn_path,
         "--method", "cnn-convert", 
         "--dry-run"], 
        "8. ANN-SNN 変換 (Dry Run)",
        # 修正: stderr も含めてチェックするため、これでパスするはず
        validator=lambda x: "SNNモデルと設定の準備が完了しました" in x
    )

    # --- 9. GraphRAG ---
    vec_store_path = "runs/health_check_vec_store"
    results["9. GraphRAG 知識追加"] = run_command(
        [python_cmd, "snn-cli.py", "knowledge", "add", 
         "HealthCheck", "System is healthy", 
         "--vector-store-path", vec_store_path],
        "9. GraphRAG 知識追加",
        validator=check_file_exists(vec_store_path) 
    )

    # --- 10. Hardware Compiler ---
    hw_config_path = "runs/health_check_hardware.yaml"
    hw_compile_code = """
from snn_research.core.snn_core import SNNCore
from snn_research.hardware.compiler import NeuromorphicCompiler
from omegaconf import OmegaConf
import logging
logging.basicConfig(level=logging.CRITICAL)
cfg = OmegaConf.load('configs/models/micro.yaml')
if 'model' in cfg: params = OmegaConf.to_container(cfg.model, resolve=True)
else: params = OmegaConf.to_container(cfg, resolve=True)
model = SNNCore(params, vocab_size=100)
compiler = NeuromorphicCompiler()
compiler.compile(model, 'runs/health_check_hardware.yaml')
"""
    results["10. ハードウェアコンパイル"] = run_command(
        [python_cmd, "-c", hw_compile_code],
        "10. ハードウェアコンパイル",
        validator=check_file_exists(hw_config_path)
    )

    # --- 11. Phase 3 Verify ---
    results["11. Phase 3 統合検証 (SFormer/SEMM)"] = run_command(
        [python_cmd, "scripts/verify_phase3.py"], 
        "11. Phase 3 統合検証 (SFormer/SEMM)",
        validator=check_log_contains("Verification Complete")
    )
    
    # --- 12. 自律エージェント ---
    results["12. 自律エージェント (Task Solver)"] = run_command(
        [python_cmd, "scripts/runners/run_agent.py", "--task_description", "health_check_task", "--force_retrain", "--model_config", "configs/models/micro.yaml", "--unlabeled_data_path", "data/smoke_test_data.jsonl"], 
        "12. 自律エージェント (Task Solver)",
        validator=check_log_contains("TASK COMPLETED")
    )

    logger.info("="*60)
    logger.info("🩺 統合健全性チェック完了 (高精度版 v2.2)")
    logger.info("="*60)
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"総合結果: {passed_count} / {total_count} のチェックを通過しました。")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status} : {name}")

    if passed_count != total_count:
        logger.error("⚠️ 一部のテストが失敗しました。ログを確認してください。")
        sys.exit(1)
    else:
        logger.info("🎉 すべてのテストが正常に完了しました。")
        sys.exit(0)

if __name__ == "__main__":
    import torch # type: ignore
    main()
