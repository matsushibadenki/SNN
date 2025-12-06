# ファイルパス: scripts/run_project_health_check.py
# Title: SNNプロジェクト 統合健全性チェック (高精度版 v2.1)
# Description: 
#   各コンポーネントの実行だけでなく、生成された成果物やログ内容を検証し、
#   システムの健全性をより厳密に診断する。
#   修正: check_log_contains で大文字小文字を無視するように変更。

import subprocess
import sys
import logging
import os
import re
import shutil
from pathlib import Path
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
        # 実行
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        # 標準出力の一部をログに記録（デバッグ用）
        if len(result.stdout) > 500:
            logger.info(f"出力抜粋: {result.stdout[:200]} ... {result.stdout[-200:]}")
        else:
            logger.info(f"出力: {result.stdout}")

        # 追加の検証ロジック
        if validator:
            if not validator(result.stdout):
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
    """指定されたファイルが存在するかチェックする"""
    def _check(_: str) -> bool:
        exists = os.path.exists(filepath)
        if not exists:
            logger.error(f"  [Check] ファイルが見つかりません: {filepath}")
        else:
            logger.info(f"  [Check] ファイル確認OK: {filepath}")
        return exists
    return _check

def check_log_contains(keyword: str, case_sensitive: bool = False) -> Callable[[str], bool]:
    """ログに指定されたキーワードが含まれているかチェックする (デフォルトは大文字小文字無視)"""
    def _check(stdout: str) -> bool:
        if case_sensitive:
            contains = keyword in stdout
        else:
            contains = keyword.lower() in stdout.lower()
            
        if not contains:
            logger.error(f"  [Check] ログにキーワード '{keyword}' が見つかりません。")
        return contains
    return _check

def check_training_success(log_dir: str) -> Callable[[str], bool]:
    """学習が正常に進行し、モデルが保存されたかチェックする"""
    def _check(stdout: str) -> bool:
        # 1. 損失がNaNになっていないか
        if "nan" in stdout.lower():
            logger.error("  [Check] 損失(Loss)が NaN になっています。")
            return False
        
        # 2. ベストモデルが存在するか
        best_model = os.path.join(log_dir, "best_model.pth")
        if not os.path.exists(best_model):
            logger.error(f"  [Check] ベストモデルが保存されていません: {best_model}")
            return False
            
        logger.info(f"  [Check] 学習正常終了確認 (Model: {best_model})")
        return True
    return _check

def main():
    logger.info("="*60)
    logger.info("🩺 SNNプロジェクト 高精度健全性チェック (v2.1)")
    logger.info("="*60)
    logger.info(f"📂 作業ディレクトリ: {os.getcwd()}")

    # 前回の実行結果をクリーンアップ（オプション）
    # shutil.rmtree("runs/smoke_tests", ignore_errors=True)

    results = {}
    python_cmd = sys.executable
    
    # --- 1. Train (Gradient) ---
    # 期待: 学習が完了し、runs/smoke_tests/best_model.pth が生成されること
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
    # 期待: "Accuracy" という単語がログに含まれること (大文字小文字無視)
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
    # 期待: 報酬履歴を含む出力ディレクトリが作成されること
    rl_out_dir = "runs/health_check_rl"
    results["3. 生物学的学習 (Bio-RL)"] = run_command(
        [python_cmd, "scripts/runners/run_rl_agent.py", 
         "--episodes", "2", 
         "--output_dir", rl_out_dir],
        "3. 生物学的学習 (Bio-RL)",
        validator=lambda x: os.path.isdir(rl_out_dir) and "Final Average Reward" in x
    )

    # --- 4. Brain Simulation ---
    # 期待: "認知サイクル完了" というログが出ること
    results["4. 認知アーキテクチャ (ArtificialBrain)"] = run_command(
        [python_cmd, "scripts/runners/run_brain_simulation.py", 
         "--prompt", "Health check prompt", 
         "--model_config", "configs/models/micro.yaml"],
        "4. 認知アーキテクチャ (ArtificialBrain)",
        validator=check_log_contains("認知サイクル完了")
    )

    # --- 5. Efficiency Report ---
    # 期待: "Total Spikes" というログが出ること
    results["5. 効率レポート (Sparsity & T)"] = run_command(
        [python_cmd, "scripts/report_sparsity_and_T.py", 
         "--model_config", "configs/models/micro.yaml", 
         "--data_path", "data/smoke_test_data.jsonl"],
        "5. 効率レポート (Sparsity & T)",
        validator=check_log_contains("Total Spikes")
    )

    # --- 6. BitNet ---
    # 期待: 学習が成功すること
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

    # エキスパート登録（依存関係のため実行、検証は簡易）
    run_command([python_cmd, "scripts/register_demo_experts.py"], "  (MoE Setup: エキスパート登録)")

    # --- 7. FrankenMoE ---
    # 期待: health_check_moe.yaml が configs/models/ に生成されること
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
    # 期待: 変換後のモデルファイルが生成されること
    conv_snn_path = "runs/converted_snn.pth"
    # ダミーANNのパス（存在チェック）
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
        validator=lambda x: "SNNモデルと設定の準備が完了しました" in x
    )

    # --- 9. GraphRAG ---
    # 期待: ベクトルストアディレクトリが生成されること
    vec_store_path = "runs/health_check_vec_store"
    results["9. GraphRAG 知識追加"] = run_command(
        [python_cmd, "snn-cli.py", "knowledge", "add", 
         "HealthCheck", "System is healthy", 
         "--vector-store-path", vec_store_path],
        "9. GraphRAG 知識追加",
        validator=check_file_exists(vec_store_path) 
    )

    # --- 10. Hardware Compiler ---
    # 期待: ハードウェア設定YAMLが生成されること
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
    
    # --- その他のデモ系 (簡易チェック) ---
    results["12. 自律エージェント (Task Solver)"] = run_command(
        [python_cmd, "scripts/runners/run_agent.py", "--task_description", "health_check_task", "--force_retrain", "--model_config", "configs/models/micro.yaml", "--unlabeled_data_path", "data/smoke_test_data.jsonl"], 
        "12. 自律エージェント (Task Solver)",
        validator=check_log_contains("TASK COMPLETED")
    )

    # 最終結果集計
    logger.info("="*60)
    logger.info("🩺 統合健全性チェック完了 (高精度版 v2.1)")
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
