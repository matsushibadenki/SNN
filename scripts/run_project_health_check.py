# ファイルパス: scripts/run_project_health_check.py
# Title: SNNプロジェクト 統合健全性チェック (最新版 v2.5 - Green AI & FF/HDC Added)
# Description: 
#   プロジェクトの全コンポーネント（学習、推論、エージェント、生物学的モデル、OS）の
#   動作を網羅的に検証するスクリプト。
#   v2.5追加: Forward-Forward学習、Hyperdimensional Computing (HDC) のテストケース。

import subprocess
import sys
import logging
import os
import shutil
import time
from typing import Callable, Optional, List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthCheckRunner:
    def __init__(self, timeout_seconds: int = 300):
        self.project_root = Path(os.getcwd())
        self.python_cmd = sys.executable
        self.timeout = timeout_seconds
        self.env = os.environ.copy()
        # プロジェクトルートをPYTHONPATHに追加して、サブプロセスでのインポートエラーを防ぐ
        self.env["PYTHONPATH"] = str(self.project_root) + os.pathsep + self.env.get("PYTHONPATH", "")

    def run_command(self, command: List[str], description: str, validator: Optional[Callable[[str], bool]] = None) -> bool:
        """
        コマンドを実行し、終了コードとバリデータによる検証を行う。
        """
        logger.info(f"--- 🏃 実行中: {description} ---")
        cmd_str = ' '.join(command)
        logger.info(f"コマンド: {cmd_str}")
        
        start_time = time.time()
        try:
            # 実行 (stdoutとstderrの両方をキャプチャ)
            # timeoutを設定してハングアップ防止
            result = subprocess.run(
                command, 
                check=True, 
                text=True, 
                capture_output=True, 
                env=self.env,
                timeout=self.timeout
            )
            
            elapsed = time.time() - start_time
            
            # --- 検証用に stdout と stderr を結合 ---
            combined_output = result.stdout + "\n" + result.stderr
            
            if result.stdout.strip():
                logger.info(f"出力(stdout) [先頭500文字]:\n{result.stdout[:500]}...")
            if result.stderr.strip():
                # 警告レベル以上の情報が含まれる可能性があるため、INFOで出力
                logger.info(f"出力(stderr) [先頭500文字]:\n{result.stderr[:500]}...")

            # 追加の検証ロジック
            if validator:
                if not validator(combined_output):
                    logger.error(f"--- ❌ 検証失敗: {description} (出力内容または成果物が期待と異なります) ---")
                    return False

            logger.info(f"--- ✅ 成功: {description} ({elapsed:.2f}s) ---")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"--- ❌ タイムアウト: {description} ({self.timeout}秒超過) ---")
            return False
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
        path = Path(filepath)
        exists = path.exists()
        if not exists:
            logger.error(f"  [Check] ファイルが見つかりません: {path}")
        else:
            logger.info(f"  [Check] ファイル確認OK: {path}")
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
        
        best_model = Path(log_dir) / "best_model.pth"
        if not best_model.exists():
            logger.error(f"  [Check] ベストモデルが保存されていません: {best_model}")
            return False
            
        logger.info(f"  [Check] 学習正常終了確認 (Model: {best_model})")
        return True
    return _check

def main():
    logger.info("="*60)
    logger.info("🩺 SNNプロジェクト 高精度健全性チェック (v2.5)")
    logger.info("="*60)
    logger.info(f"📂 作業ディレクトリ: {os.getcwd()}")

    runner = HealthCheckRunner(timeout_seconds=600) # 学習を含むため長めに設定
    results = {}
    
    # Pythonコマンドのショートカット
    py = runner.python_cmd

    # 一時ファイル/ディレクトリの管理リスト（最後に削除を試みる）
    temp_paths = []

    try:
        # --- 1. Train ---
        log_dir_1 = "runs/smoke_tests"
        temp_paths.append(log_dir_1)
        results["1. 代理勾配学習 (gradient_based)"] = runner.run_command(
            [py, "scripts/runners/train.py", 
             "--config", "configs/experiments/smoke_test_config.yaml",
             "--model_config", "configs/models/micro.yaml", 
             "--paradigm", "gradient_based", 
             "--data_path", "data/smoke_test_data.jsonl",
             "--override_config", f"training.log_dir={log_dir_1}"],
            "1. 代理勾配学習 (gradient_based)",
            validator=check_training_success(log_dir_1)
        )

        # --- 2. Benchmark ---
        results["2. ベンチマーク実行 (Train+Eval)"] = runner.run_command(
            [py, "scripts/run_benchmark_suite.py", 
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
        temp_paths.append(rl_out_dir)
        results["3. 生物学的学習 (Bio-RL)"] = runner.run_command(
            [py, "scripts/runners/run_rl_agent.py", 
             "--episodes", "2", 
             "--output_dir", rl_out_dir],
            "3. 生物学的学習 (Bio-RL)",
            validator=lambda x: Path(rl_out_dir).is_dir() and "Final Average Reward" in x
        )

        # --- 4. Brain Simulation ---
        results["4. 認知アーキテクチャ (ArtificialBrain)"] = runner.run_command(
            [py, "scripts/runners/run_brain_simulation.py", 
             "--prompt", "Health check prompt", 
             "--model_config", "configs/models/micro.yaml"],
            "4. 認知アーキテクチャ (ArtificialBrain)",
            validator=check_log_contains("認知サイクル完了")
        )

        # --- 5. Efficiency Report ---
        results["5. 効率レポート (Sparsity & T)"] = runner.run_command(
            [py, "scripts/report_sparsity_and_T.py", 
             "--model_config", "configs/models/micro.yaml", 
             "--data_path", "data/smoke_test_data.jsonl"],
            "5. 効率レポート (Sparsity & T)",
            validator=check_log_contains("Total Spikes")
        )

        # --- 6. BitNet ---
        bitnet_log_dir = "runs/smoke_tests_bitnet"
        temp_paths.append(bitnet_log_dir)
        results["6. 1.58bit BitNet学習 (BitRWKV)"] = runner.run_command(
            [py, "scripts/runners/train.py", 
             "--config", "configs/experiments/smoke_test_config.yaml",
             "--model_config", "configs/models/bit_rwkv_micro.yaml", 
             "--paradigm", "gradient_based", 
             "--data_path", "data/smoke_test_data.jsonl",
             "--override_config", "training.epochs=1",
             "--override_config", f"training.log_dir={bitnet_log_dir}"],
            "6. 1.58bit BitNet学習 (BitRWKV)",
            validator=check_training_success(bitnet_log_dir)
        )

        # エキスパート登録 (前提条件)
        runner.run_command([py, "scripts/register_demo_experts.py"], "  (MoE Setup: エキスパート登録)")

        # --- 7. FrankenMoE ---
        moe_config_name = "health_check_moe.yaml"
        moe_config_path = Path("configs") / "models" / moe_config_name
        temp_paths.append(str(moe_config_path))
        
        # 既存があれば削除
        if moe_config_path.exists():
            moe_config_path.unlink()

        results["7. FrankenMoE 構成ビルド"] = runner.run_command(
            [py, "scripts/manage_models.py", "build-moe", 
             "--keywords", "science,history", 
             "--output", moe_config_name],
            "7. FrankenMoE 構成ビルド",
            validator=check_file_exists(str(moe_config_path))
        )

        # --- 8. ANN-SNN Conversion ---
        conv_snn_path = "runs/converted_snn.pth"
        dummy_ann_path = "runs/dummy_ann.pth"
        temp_paths.append(conv_snn_path)
        temp_paths.append(dummy_ann_path)

        # ダミーANNモデルの作成
        if not os.path.exists(dummy_ann_path):
            import torch
            import torch.nn as nn
            # ディレクトリ作成
            os.makedirs(os.path.dirname(dummy_ann_path), exist_ok=True)
            torch.save(nn.Linear(10, 10).state_dict(), dummy_ann_path)

        results["8. ANN-SNN 変換 (Dry Run)"] = runner.run_command(
            [py, "scripts/convert_model.py", 
             "--ann_model_path", dummy_ann_path, 
             "--snn_model_config", "configs/models/micro.yaml", 
             "--output_snn_path", conv_snn_path,
             "--method", "cnn-convert", 
             "--dry-run"], 
            "8. ANN-SNN 変換 (Dry Run)",
            validator=lambda x: "SNNモデルと設定の準備が完了しました" in x
        )

        # --- 9. GraphRAG ---
        vec_store_path = "runs/health_check_vec_store"
        temp_paths.append(vec_store_path)
        results["9. GraphRAG 知識追加"] = runner.run_command(
            [py, "snn-cli.py", "knowledge", "add", 
             "HealthCheck", "System is healthy", 
             "--vector-store-path", vec_store_path],
            "9. GraphRAG 知識追加",
            validator=check_file_exists(vec_store_path) 
        )

        # --- 10. Hardware Compiler ---
        hw_config_path = "runs/health_check_hardware.yaml"
        temp_paths.append(hw_config_path)
        
        # インラインスクリプトでのコンパイルテスト
        hw_compile_code = """
from snn_research.core.snn_core import SNNCore
from snn_research.hardware.compiler import NeuromorphicCompiler
from omegaconf import OmegaConf
import logging
import sys
# ロガーを抑制
logging.basicConfig(level=logging.CRITICAL)
try:
    cfg = OmegaConf.load('configs/models/micro.yaml')
    if 'model' in cfg: params = OmegaConf.to_container(cfg.model, resolve=True)
    else: params = OmegaConf.to_container(cfg, resolve=True)
    model = SNNCore(params, vocab_size=100)
    compiler = NeuromorphicCompiler()
    compiler.compile(model, 'runs/health_check_hardware.yaml')
except Exception as e:
    print(f"Compile Error: {e}", file=sys.stderr)
    sys.exit(1)
"""
        results["10. ハードウェアコンパイル"] = runner.run_command(
            [py, "-c", hw_compile_code],
            "10. ハードウェアコンパイル",
            validator=check_file_exists(hw_config_path)
        )

        # --- 11. Phase 3 Verify ---
        results["11. Phase 3 統合検証 (SFormer/SEMM)"] = runner.run_command(
            [py, "scripts/verify_phase3.py"], 
            "11. Phase 3 統合検証 (SFormer/SEMM)",
            validator=check_log_contains("Verification Complete")
        )
        
        # --- 12. 自律エージェント ---
        results["12. 自律エージェント (Task Solver)"] = runner.run_command(
            [py, "scripts/runners/run_agent.py", 
             "--task_description", "health_check_task", 
             "--force_retrain", 
             "--model_config", "configs/models/micro.yaml", 
             "--unlabeled_data_path", "data/smoke_test_data.jsonl"], 
            "12. 自律エージェント (Task Solver)",
            validator=check_log_contains("TASK COMPLETED")
        )

        # --- 13. On-Chip Learning ---
        results["13. オンチップ学習 (STDP)"] = runner.run_command(
            [py, "scripts/run_on_chip_learning.py"],
            "13. オンチップ学習 (STDP)",
            validator=check_log_contains("demo finished")
        )

        # --- 14. Bio-Microcircuit (PD14) ---
        results["14. 生物学的マイクロサーキット (PD14)"] = runner.run_command(
            [py, "scripts/run_bio_microcircuit_demo.py"],
            "14. 生物学的マイクロサーキット (PD14)",
            validator=check_log_contains("Demo Completed")
        )

        # --- 15. Neuromorphic OS ---
        results["15. 脳型OSシミュレーション (Neuromorphic OS)"] = runner.run_command(
            [py, "scripts/runners/run_neuromorphic_os.py"],
            "15. 脳型OSシミュレーション (Neuromorphic OS)",
            validator=check_log_contains("Demo Completed")
        )

        # --- 16. Forward-Forward Learning (New) ---
        # インラインスクリプトでFFトレーナーの動作確認を行う
        ff_check_code = """
import torch
import torch.nn as nn
from snn_research.training.trainers.forward_forward import ForwardForwardTrainer
import sys

try:
    # 簡易モデルの定義 (Sequential)
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
    )
    
    # ダミーデータ生成
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4,))
    # DataLoader形式のリスト
    loader = [(inputs, targets)]
    
    # トレーナー初期化と学習実行
    trainer = ForwardForwardTrainer(model, learning_rate=0.01)
    metrics = trainer.train_epoch(loader)
    
    print(f"FF Training Finished. Metrics: {metrics}")
    
    if 'loss' not in metrics:
        raise ValueError("Metrics do not contain loss")
        
except Exception as e:
    print(f"FF Error: {e}", file=sys.stderr)
    sys.exit(1)
"""
        results["16. Forward-Forward (FF) 学習"] = runner.run_command(
            [py, "-c", ff_check_code],
            "16. Forward-Forward (FF) 学習",
            validator=check_log_contains("FF Training Finished")
        )

        # --- 17. Hyperdimensional Computing (HDC) (New) ---
        hdc_check_code = """
from snn_research.cognitive_architecture.hdc_engine import HDCEngine, HDCReasoningAgent
import sys

try:
    # テスト用に低次元で初期化
    engine = HDCEngine(dim=2048) 
    agent = HDCReasoningAgent(engine)
    
    # シンボル接地と推論テスト: Japan + Capital -> Tokyo
    agent.learn_concept("Japan", "Capital", "Tokyo")
    result = agent.query("Japan", "Capital")
    
    print(f"HDC Query Result: {result}")
    
    # 推論結果の検証
    if not result or result[0][0] != "Tokyo":
        raise Exception(f"Inference failed. Expected Tokyo, got {result}")
        
    print("HDC Test Passed")

except Exception as e:
    print(f"HDC Error: {e}", file=sys.stderr)
    sys.exit(1)
"""
        results["17. Hyperdimensional Computing (HDC)"] = runner.run_command(
            [py, "-c", hdc_check_code],
            "17. Hyperdimensional Computing (HDC)",
            validator=check_log_contains("HDC Test Passed")
        )

    finally:
        # 結果集計
        logger.info("="*60)
        logger.info("🩺 統合健全性チェック完了 (高精度版 v2.5)")
        logger.info("="*60)
        
        passed_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        logger.info(f"総合結果: {passed_count} / {total_count} のチェックを通過しました。")
        for name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"  {status} : {name}")
            
        # Cleanup
        # for p in temp_paths:
        #     try:
        #         if os.path.isfile(p): os.remove(p)
        #         elif os.path.isdir(p): shutil.rmtree(p)
        #     except Exception: pass

    if passed_count != total_count:
        logger.error("⚠️ 一部のテストが失敗しました。ログを確認してください。")
        sys.exit(1)
    else:
        logger.info("🎉 すべてのテストが正常に完了しました。")
        sys.exit(0)

if __name__ == "__main__":
    main()
