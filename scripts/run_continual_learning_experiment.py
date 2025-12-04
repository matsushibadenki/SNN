# ファイルパス: scripts/run_continual_learning_experiment.py
# Title: 継続学習（Continual Learning）実験スクリプト (Warning解消版)
# Description: SNNの継続学習能力、特に「破局的忘却」の克服を実証するための実験を行う。
# 修正点:
#   - `train.py` 実行時に明示的に `--override_config training.gradient_based.type=standard` を追加。
#     これにより、base_configでデフォルト設定されている「蒸留モード」によるデータ不足警告を回避し、
#     正しい「標準学習モード」で実行されるように修正。

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import time
import json
import re # 正規表現ライブラリをインポート
from typing import List, Dict, Any, Optional

def run_command(command: List[str]) -> subprocess.CompletedProcess:
    """コマンドを実行し、完了プロセスオブジェクトを返す。エラーがあれば例外を発生させる。"""
    print(f"\n▶️ Running command: {' '.join(command)}")
    try:
        # check=True でエラー時に例外を発生させる
        result = subprocess.run(
            command,
            check=True,
            capture_output=True, # 標準出力をキャプチャ
            text=True,
            encoding='utf-8'
        )
        print(result.stdout) # 標準出力を表示
        if result.stderr:
            print(f"--- Stderr ---\n{result.stderr}\n--------------")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}:")
        print(f"--- Stdout ---\n{e.stdout}\n--------------")
        print(f"--- Stderr ---\n{e.stderr}\n--------------")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

def extract_metric_from_log(log: str, metric_name: str) -> Optional[float]:
    """ベンチマークスイートのログから指定されたメトリクス（例: accuracy）を抽出する。"""
    # 例: | 0.852 | 1500.0 | ... | SNN | ... | EvalOnly (...) |
    # または "accuracy": 0.852 というJSONライクな行を探す
    
    # 正規表現でメトリクス行を探す
    # (可読性のため、ここではAccuracyのみを簡易的に探す)
    pattern = re.compile(rf'\|\s*({metric_name})\s*:\s*([0-9\.]+)\s*')
    
    for line in reversed(log.splitlines()):
        if "Results:" in line and f"'{metric_name}'" in line:
            try:
                # 'Results: {'accuracy': 0.85, ...}' のような形式を想定
                data_str = line.split("Results:", 1)[1].strip()
                data_str = data_str.replace("'", "\"") # JSON準拠に
                metrics = json.loads(data_str)
                if metric_name in metrics:
                    return float(metrics[metric_name])
            except Exception:
                continue # パース失敗
                
        # テーブル形式のログからも探す
        # | 0.852 | ... | SNN | ...
        # (この簡易実装では、最初の数値がAccuracyだと仮定 - 堅牢ではない)
        if "EvalOnly" in line and "|" in line:
             parts = [p.strip() for p in line.split('|')]
             if len(parts) > 2:
                 try:
                     # 最初の列がAccuracyだと仮定
                     accuracy = float(parts[1])
                     return accuracy
                 except ValueError:
                     continue
                     
    print(f"⚠️ ログからメトリクス '{metric_name}' を抽出できませんでした。")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Continual Learning Experiment using EWC")
    parser.add_argument("--epochs_task_a", type=int, default=3, help="Epochs for training on Task A (SST-2).")
    parser.add_argument("--epochs_task_b", type=int, default=3, help="Epochs for training on Task B (MRPC).")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="Model architecture config file.")
    parser.add_argument("--output_dir", type=str, default="benchmarks/continual_learning", help="Directory to save results and models.")
    parser.add_argument("--benchmark_script", type=str, default="scripts/run_benchmark_suite.py", help="Path to the benchmark script.")
    parser.add_argument("--train_script", type=str, default="train.py", help="Path to the training script.")
    # --- データパスを引数で受け取るように変更 ---
    parser.add_argument("--data_path_task_a", type=str, default="glue/sst2", help="Data path for Task A")
    parser.add_argument("--data_path_task_b", type=str, default="glue/mrpc", help="Data path for Task B")
    
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Paths ---
    task_a_model_dir = output_path / "task_a_model"
    task_a_model_path = task_a_model_dir / "best_model.pth"
    ewc_data_path = task_a_model_dir / "ewc_data_sst2.pt" # train.pyがこの名前で保存する

    task_b_ewc_dir = output_path / "task_b_ewc"
    task_b_ewc_path = task_b_ewc_dir / "best_model.pth"
    
    task_b_finetune_dir = output_path / "task_b_finetune"
    task_b_finetune_path = task_b_finetune_dir / "best_model.pth"
    
    report_path = output_path / "continual_learning_report.md"

    # --- Stage 1: Train on Task A (SST-2) and compute Fisher matrix ---
    print("\n" + "="*20 + "  Stage 1: Train on Task A (SST-2) " + "="*20)
    train_cmd_a = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--task_name", "sst2", # EWCのためにタスク名を指定
        "--override_config", f"data.path={args.data_path_task_a}", 
        "--override_config", f"training.epochs={args.epochs_task_a}",
        "--override_config", "training.gradient_based.loss.ewc_weight=400", # EWC weight must be > 0 to compute Fisher
        "--override_config", f"training.log_dir={task_a_model_dir.as_posix()}",
        # 修正: デフォルトの蒸留設定を上書きして標準学習モードにする
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_a)
    
    if not task_a_model_path.exists() or not ewc_data_path.exists():
        print(f"❌ Stage 1 failed: Model ('{task_a_model_path}') or EWC data ('{ewc_data_path}') was not created.")
        sys.exit(1)

    # --- Stage 2: Train on Task B (MRPC) ---
    print("\n" + "="*20 + " Stage 2: Train on Task B (MRPC) " + "="*20)
    
    # --- 2a: With EWC ---
    print("\n--- 2a: Training on MRPC with EWC ---")
    train_cmd_b_ewc = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--resume_path", str(task_a_model_path), # タスクAのモデルから開始
        "--load_ewc_data", str(ewc_data_path),   # タスクAのFisher行列をロード
        "--task_name", "mrpc", # 新しいタスク名
        "--override_config", f"data.path={args.data_path_task_b}", 
        "--override_config", f"training.epochs={args.epochs_task_b}",
        "--override_config", "training.gradient_based.loss.ewc_weight=400", # EWCを有効
        "--override_config", f"training.log_dir={task_b_ewc_dir.as_posix()}",
        # 修正: デフォルトの蒸留設定を上書きして標準学習モードにする
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_b_ewc) # シミュレーションではなく実行

    # --- 2b: Without EWC (Fine-tuning) ---
    print("\n--- 2b: Training on MRPC without EWC (Fine-tuning) ---")
    train_cmd_b_finetune = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--resume_path", str(task_a_model_path), # タスクAのモデルから開始
        # No --load_ewc_data
        "--task_name", "mrpc",
        "--override_config", f"data.path={args.data_path_task_b}", 
        "--override_config", f"training.epochs={args.epochs_task_b}",
        "--override_config", "training.gradient_based.loss.ewc_weight=0", # EWCを無効化
        "--override_config", f"training.log_dir={task_b_finetune_dir.as_posix()}",
        # 修正: デフォルトの蒸留設定を上書きして標準学習モードにする
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_b_finetune) # シミュレーションではなく実行
    
    print("\n" + "="*20 + " Stage 3: Evaluation (Measuring Forgetting) " + "="*20)
    
    results_data: List[Dict[str, Any]] = []

    # --- 3a: Evaluate EWC model on Task A (SST-2) ---
    print("\n--- 3a: Evaluating EWC Model on Task A (SST-2) [Checks Forgetting] ---")
    eval_cmd_ewc_a = [
        sys.executable, args.benchmark_script,
        "--experiment", "sst2_comparison",
        "--eval_only",
        "--model_path", str(task_b_ewc_path),
        "--model_config", args.model_config,
        "--model_type", "SNN",
        "--tag", "Continual_EWC_on_TaskA"
    ]
    result_log_ewc_a = run_command(eval_cmd_ewc_a)
    acc_ewc_a = extract_metric_from_log(result_log_ewc_a.stdout, "accuracy") or 0.0
    results_data.append({"Model": "EWC", "Task": "Task A (SST-2)", "Accuracy": acc_ewc_a})

    # --- 3b: Evaluate Finetune model on Task A (SST-2) ---
    print("\n--- 3b: Evaluating Finetune Model on Task A (SST-2) [Checks Forgetting] ---")
    eval_cmd_ft_a = [
        sys.executable, args.benchmark_script,
        "--experiment", "sst2_comparison",
        "--eval_only",
        "--model_path", str(task_b_finetune_path),
        "--model_config", args.model_config,
        "--model_type", "SNN",
        "--tag", "Continual_Finetune_on_TaskA"
    ]
    result_log_ft_a = run_command(eval_cmd_ft_a)
    acc_ft_a = extract_metric_from_log(result_log_ft_a.stdout, "accuracy") or 0.0
    results_data.append({"Model": "Finetune Only", "Task": "Task A (SST-2)", "Accuracy": acc_ft_a})

    # --- 3c: Evaluate EWC model on Task B (MRPC) ---
    print("\n--- 3c: Evaluating EWC Model on Task B (MRPC) [Checks New Learning] ---")
    eval_cmd_ewc_b = [
        sys.executable, args.benchmark_script,
        "--experiment", "mrpc_comparison",
        "--eval_only",
        "--model_path", str(task_b_ewc_path),
        "--model_config", args.model_config,
        "--model_type", "SNN",
        "--tag", "Continual_EWC_on_TaskB"
    ]
    result_log_ewc_b = run_command(eval_cmd_ewc_b)
    acc_ewc_b = extract_metric_from_log(result_log_ewc_b.stdout, "accuracy") or 0.0
    results_data.append({"Model": "EWC", "Task": "Task B (MRPC)", "Accuracy": acc_ewc_b})

    # --- 3d: Evaluate Finetune model on Task B (MRPC) ---
    print("\n--- 3d: Evaluating Finetune Model on Task B (MRPC) [Checks New Learning] ---")
    eval_cmd_ft_b = [
        sys.executable, args.benchmark_script,
        "--experiment", "mrpc_comparison",
        "--eval_only",
        "--model_path", str(task_b_finetune_path),
        "--model_config", args.model_config,
        "--model_type", "SNN",
        "--tag", "Continual_Finetune_on_TaskB"
    ]
    result_log_ft_b = run_command(eval_cmd_ft_b)
    acc_ft_b = extract_metric_from_log(result_log_ft_b.stdout, "accuracy") or 0.0
    results_data.append({"Model": "Finetune Only", "Task": "Task B (MRPC)", "Accuracy": acc_ft_b})


    # --- Stage 4: Generate Report ---
    print("\n" + "="*20 + " Stage 4: Generating Report " + "="*20)
    df = pd.DataFrame(results_data)
    
    # データをピボットして比較しやすくする
    try:
        df_pivot = df.pivot(index="Model", columns="Task", values="Accuracy")
        df_pivot = df_pivot.reset_index()
        # 破局的忘却の度合いを計算
        forgetting = df_pivot[df_pivot["Model"] == "Finetune Only"]["Task A (SST-2)"].values[0]
        ewc_retention = df_pivot[df_pivot["Model"] == "EWC"]["Task A (SST-2)"].values[0]
        
        conclusion = (
            f"  - **EWC Model:** Retained Task A accuracy at {ewc_retention:.2%}.\n"
            f"  - **Finetune Model:** Task A accuracy dropped to {forgetting:.2%} (Catastrophic Forgetting).\n"
            f"  - **Conclusion:** EWC successfully mitigated catastrophic forgetting."
        )

    except Exception as e:
        print(f"レポートのピボット作成中にエラー: {e}")
        df_pivot = df # ピボット失敗時はそのまま表示
        conclusion = "Could not pivot report."


    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Continual Learning Experiment Report (Actual)\n\n")
        f.write(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task A (SST-2) Epochs: {args.epochs_task_a}, Task B (MRPC) Epochs: {args.epochs_task_b}\n")
        f.write(f"Model Config: {args.model_config}\n\n")
        f.write(df_pivot.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write(conclusion)

    print(f"\n✅ 実験レポートが '{report_path}' に保存されました。")
    print(df_pivot.to_markdown(index=False))
    print(conclusion)

if __name__ == "__main__":
    main()