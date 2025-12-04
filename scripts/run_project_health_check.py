# ファイルパス: scripts/run_project_health_check.py
#
# Title: SNNプロジェクト 統合健全性チェック (Enhanced v2.1)
#
# Description:
# プロジェクトの全フェーズ（Phase 1～3+）にわたる主要機能の動作を網羅的に検証するスクリプト。
# 学習、推論、エージェント、新アーキテクチャ、デモ、ツール群の健全性を診断する。


import subprocess
import sys
import os
import logging
import argparse

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """
    指定されたコマンドを実行し、結果をログに記録します。
    """
    logger.info(f"--- 🏃 実行中: {description} ---")
    logger.info(f"コマンド: {command}")
    
    try:
        # コマンドをリスト形式に分割（簡易的な処理）
        # 注意: 複雑な引数（引用符など）がある場合は shlex.split 推奨だが、ここでは簡易実装
        import shlex
        args = shlex.split(command)
        
        # サブプロセス実行
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 成功時
        # 出力が多い場合は先頭/末尾だけ表示するなどの工夫も可能だが、
        # ここではエラー時のみ詳細を出す方針、あるいは重要なログはinfoで出す
        if result.stdout:
            # ログが多すぎると見づらいので、キーワードを含む行や最後の数行だけ出すなど調整
            # ここではシンプルに成功メッセージ
            pass
            
        logger.info(f"--- ✅ 成功: {description} ---")
        return True, result.stdout

    except subprocess.CalledProcessError as e:
        logger.error(f"--- ❌ 失敗: {description} (終了コード: {e.returncode}) ---")
        # エラー出力があれば表示
        if e.stderr:
            logger.info(f"  [{description}] エラー出力:\n{e.stderr}")
        if e.stdout:
            logger.info(f"  [{description}] 標準出力（参考）:\n{e.stdout}")
        return False, e.stderr
    except FileNotFoundError:
        logger.error(f"--- ❌ 失敗: {description} (コマンドが見つかりません) ---")
        return False, "Command not found"

def main():
    parser = argparse.ArgumentParser(description='SNNプロジェクト統合健全性チェック')
    args = parser.parse_args()

    project_root = os.getcwd()
    logger.info("============================================================")
    logger.info("🩺 SNNプロジェクト 統合健全性チェック (Runners Path Fixed)")
    logger.info("============================================================")
    logger.info(f"📂 作業ディレクトリ: {project_root}")

    python_executable = sys.executable

    # チェック対象のコマンドリスト
    # 実行ファイルのパスを scripts/runners/ 配下に修正
    commands = [
        # 1. 基本的な学習ループ (Gradient Based)
        {
            "desc": "1. 代理勾配学習 (gradient_based)",
            "cmd": f"{python_executable} scripts/runners/train.py --config configs/experiments/smoke_test_config.yaml --model_config configs/models/micro.yaml --paradigm gradient_based --data_path data/smoke_test_data.jsonl"
        },
        # 2. ベンチマークスイート (Train + Eval)
        {
            "desc": "2. ベンチマーク実行 (Train+Eval)",
            "cmd": f"{python_executable} scripts/run_benchmark_suite.py --experiment health_check_comparison --epochs 1 --batch_size 2 --model_config configs/models/micro.yaml --tag HealthCheck"
        },
        # 3. 生物学的学習 (Bio-RL)
        {
            "desc": "3. 生物学的学習 (Bio-RL)",
            "cmd": f"{python_executable} scripts/runners/run_rl_agent.py --episodes 2 --output_dir runs/health_check_rl"
        },
        # 4. 認知アーキテクチャ (Artificial Brain)
        {
            "desc": "4. 認知アーキテクチャ (ArtificialBrain)",
            "cmd": f"{python_executable} scripts/runners/run_brain_simulation.py --prompt \"Health check prompt\" --model_config configs/models/micro.yaml"
        },
        # 5. 効率性レポート (Sparsity & T)
        {
            "desc": "5. 効率レポート (Sparsity & T)",
            "cmd": f"{python_executable} scripts/report_sparsity_and_T.py --model_config configs/models/micro.yaml --data_path data/smoke_test_data.jsonl"
        },
        # 6. BitNet (1.58bit) 学習
        {
            "desc": "6. 1.58bit BitNet学習 (BitRWKV)",
            "cmd": f"{python_executable} scripts/runners/train.py --config configs/experiments/smoke_test_config.yaml --model_config configs/models/bit_rwkv_micro.yaml --paradigm gradient_based --data_path data/smoke_test_data.jsonl --override_config training.epochs=1"
        },
        # --- ここから下はパス変更の影響が少ないスクリプト群 ---
        {
            "desc": "  (MoE Setup: エキスパート登録)",
            "cmd": f"{python_executable} scripts/register_demo_experts.py"
        },
        {
            "desc": "7. FrankenMoE 構成ビルド",
            "cmd": f"{python_executable} scripts/manage_models.py build-moe --keywords science,history --output health_check_moe.yaml"
        },
        {
            "desc": "8. ANN-SNN 変換 (Dry Run)",
            "cmd": f"{python_executable} scripts/convert_model.py --ann_model_path runs/dummy_ann.pth --snn_model_config configs/models/micro.yaml --output_snn_path runs/converted_snn.pth --method cnn-convert --dry-run"
        },
        {
            "desc": "9. GraphRAG 知識追加",
            "cmd": f"{python_executable} snn-cli.py knowledge add \"HealthCheck\" \"System is healthy\" --vector-store-path runs/health_check_vec_store"
        },
        # 10. ハードウェアコンパイラ (Pythonコードとして実行)
        {
            "desc": "10. ハードウェアコンパイル",
            "cmd": f"""{python_executable} -c "
from snn_research.core.snn_core import SNNCore
from snn_research.hardware.compiler import NeuromorphicCompiler
from omegaconf import OmegaConf
import logging
logging.basicConfig(level=logging.CRITICAL)
cfg = OmegaConf.load('configs/models/micro.yaml')
# OmegaConfオブジェクトを通常のDictに変換して渡す
model = SNNCore(OmegaConf.to_container(cfg, resolve=True), vocab_size=100)
compiler = NeuromorphicCompiler()
compiler.compile(model, 'runs/health_check_hardware.yaml')
" """
        },
        {
            "desc": "11. Phase 3 統合検証 (SFormer/SEMM)",
            "cmd": f"{python_executable} scripts/verify_phase3.py"
        },
        # --- 再び実行スクリプト系 ---
        {
            "desc": "12. 自律エージェント (Task Solver)",
            "cmd": f"{python_executable} scripts/runners/run_agent.py --task_description health_check_task --force_retrain --model_config configs/models/micro.yaml --unlabeled_data_path data/smoke_test_data.jsonl"
        },
        {
            "desc": "13. 階層的プランナー (Planner)",
            "cmd": f"{python_executable} scripts/runners/run_planner.py --task_request \"Plan a health check sequence\" --context_data \"System status is unknown\""
        },
        {
            "desc": "14. マルチモーダル処理デモ",
            "cmd": f"{python_executable} scripts/run_multimodal_demo.py"
        },
        {
            "desc": "15. 空間認識デモ",
            "cmd": f"{python_executable} scripts/run_spatial_demo.py"
        },
        {
            "desc": "16. ECG異常検知デモ",
            "cmd": f"{python_executable} scripts/run_ecg_analysis.py --model_config configs/models/ecg_temporal_snn.yaml --num_samples 2 --time_steps 100"
        },
        {
            "desc": "17. ハイパーパラメータ最適化 (HPO)",
            "cmd": f"{python_executable} scripts/runners/run_hpo.py --model_config configs/models/micro.yaml --task health_check --n_trials 1 --eval_epochs 1 --metric_name accuracy --output_base_dir runs/health_check_hpo"
        },
        {
            "desc": "18. スパイク活動可視化 (Patterns)",
            "cmd": f"{python_executable} scripts/visualize_spike_patterns.py --model_config configs/models/micro.yaml --timesteps 8 --batch_size 2 --output_prefix runs/health_check_viz"
        },
        {
            "desc": "19. ニューロンダイナミクス可視化",
            "cmd": f"{python_executable} scripts/visualize_neuron_dynamics.py --model_config configs/models/micro.yaml --timesteps 8 --output_path runs/health_check_dynamics.png"
        }
    ]

    success_count = 0
    fail_count = 0
    results = []

    for item in commands:
        success, output = run_command(item["cmd"], item["desc"])
        if success:
            success_count += 1
            results.append((item["desc"], "✅ PASS"))
        else:
            fail_count += 1
            results.append((item["desc"], "❌ FAIL"))

    logger.info("============================================================")
    logger.info("🩺 統合健全性チェック完了")
    logger.info("============================================================")
    logger.info(f"総合結果: {success_count} / {len(commands)} の主要機能が正常に動作しました。")
    
    for desc, status in results:
        logger.info(f"  {status} : {desc}")

    if fail_count > 0:
        logger.error(f"⚠️ {fail_count} 個のチェックが失敗しました。詳細は上記のログを確認してください。")
        sys.exit(1)
    else:
        logger.info("🎉 全てのチェックが通過しました！プロジェクトは健全です。")
        sys.exit(0)

if __name__ == "__main__":
    main()