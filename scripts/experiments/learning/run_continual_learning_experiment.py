# ファイルパス: scripts/run_continual_learning_experiment.py
# (修正: train_script のデフォルトパスを修正)

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import json

from typing import List, Dict, Any, Optional


def run_command(command: List[str]) -> subprocess.CompletedProcess:
    # ... (既存コードと同じ)
    print(f"\n▶️ Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
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
    # ... (既存コードと同じ)

    for line in reversed(log.splitlines()):
        if "Results:" in line and f"'{metric_name}'" in line:
            try:
                data_str = line.split("Results:", 1)[
                    1].strip().replace("'", "\"")
                metrics = json.loads(data_str)
                if metric_name in metrics:
                    return float(metrics[metric_name])
            except Exception:
                continue
        if "EvalOnly" in line and "|" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNN Continual Learning Experiment using EWC")
    parser.add_argument("--epochs_task_a", type=int, default=3)
    parser.add_argument("--epochs_task_b", type=int, default=3)
    parser.add_argument("--model_config", type=str,
                        default="configs/models/small.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="workspace/benchmarks/continual_learning")
    parser.add_argument("--benchmark_script", type=str,
                        default="scripts/run_benchmark_suite.py")

    # --- ▼ 修正: デフォルトパスを scripts/runners/train.py に変更 ▼ ---
    parser.add_argument("--train_script", type=str,
                        default="scripts/runners/train.py")
    # --- ▲ 修正 ▲ ---

    parser.add_argument("--data_path_task_a", type=str, default="glue/sst2")
    parser.add_argument("--data_path_task_b", type=str, default="glue/mrpc")
    args = parser.parse_args()

    # ... (以下、既存コードと同じロジック)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Paths ---
    task_a_model_dir = output_path / "task_a_model"
    task_a_model_path = task_a_model_dir / "best_model.pth"
    ewc_data_path = task_a_model_dir / "ewc_data_sst2.pt"

    task_b_ewc_dir = output_path / "task_b_ewc"
    task_b_ewc_path = task_b_ewc_dir / "best_model.pth"

    task_b_finetune_dir = output_path / "task_b_finetune"
    task_b_finetune_path = task_b_finetune_dir / "best_model.pth"

    report_path = output_path / "continual_learning_report.md"

    # --- Stage 1 ---
    print("\n" + "="*20 + "  Stage 1: Train on Task A (SST-2) " + "="*20)
    train_cmd_a = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--task_name", "sst2",
        "--override_config", f"data.path={args.data_path_task_a}",
        "--override_config", f"training.epochs={args.epochs_task_a}",
        "--override_config", "training.gradient_based.loss.ewc_weight=400",
        "--override_config", f"training.log_dir={task_a_model_dir.as_posix()}",
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_a)

    if not task_a_model_path.exists() or not ewc_data_path.exists():
        print("❌ Stage 1 failed.")
        sys.exit(1)

    # --- Stage 2 ---
    print("\n" + "="*20 + " Stage 2: Train on Task B (MRPC) " + "="*20)

    print("\n--- 2a: With EWC ---")
    train_cmd_b_ewc = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--resume_path", str(task_a_model_path),
        "--load_ewc_data", str(ewc_data_path),
        "--task_name", "mrpc",
        "--override_config", f"data.path={args.data_path_task_b}",
        "--override_config", f"training.epochs={args.epochs_task_b}",
        "--override_config", "training.gradient_based.loss.ewc_weight=400",
        "--override_config", f"training.log_dir={task_b_ewc_dir.as_posix()}",
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_b_ewc)

    print("\n--- 2b: Without EWC ---")
    train_cmd_b_finetune = [
        sys.executable, args.train_script,
        "--model_config", args.model_config,
        "--resume_path", str(task_a_model_path),
        "--task_name", "mrpc",
        "--override_config", f"data.path={args.data_path_task_b}",
        "--override_config", f"training.epochs={args.epochs_task_b}",
        "--override_config", "training.gradient_based.loss.ewc_weight=0",
        "--override_config", f"training.log_dir={task_b_finetune_dir.as_posix()}",
        "--override_config", "training.gradient_based.type=standard"
    ]
    run_command(train_cmd_b_finetune)

    # --- Stage 3 ---
    print("\n" + "="*20 + " Stage 3: Evaluation " + "="*20)
    results_data: List[Dict[str, Any]] = []

    for model_type, model_path, tag in [("EWC", task_b_ewc_path, "EWC"), ("Finetune Only", task_b_finetune_path, "Finetune")]:
        for task, task_tag in [("Task A (SST-2)", "TaskA"), ("Task B (MRPC)", "TaskB")]:
            print(f"\n--- Evaluating {model_type} on {task} ---")
            cmd = [
                sys.executable, args.benchmark_script,
                "--experiment", "sst2_comparison" if "A" in task_tag else "mrpc_comparison",
                "--eval_only",
                "--model_path", str(model_path),
                "--model_config", args.model_config,
                "--model_type", "SNN",
                "--tag", f"{tag}_on_{task_tag}"
            ]
            log = run_command(cmd).stdout
            acc = extract_metric_from_log(log, "accuracy") or 0.0
            results_data.append(
                {"Model": model_type, "Task": task, "Accuracy": acc})

    # --- Stage 4 ---
    print("\n" + "="*20 + " Stage 4: Report " + "="*20)
    df = pd.DataFrame(results_data)
    try:
        df_pivot = df.pivot(index="Model", columns="Task",
                            values="Accuracy").reset_index()
        forgetting = df_pivot[df_pivot["Model"] ==
                              "Finetune Only"]["Task A (SST-2)"].values[0]
        ewc_retention = df_pivot[df_pivot["Model"]
                                 == "EWC"]["Task A (SST-2)"].values[0]
        conclusion = f"  - EWC: {ewc_retention:.2%}, Finetune: {forgetting:.2%} (Task A Accuracy)"
    except Exception:
        df_pivot = df
        conclusion = "Pivot failed."

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(
            f"# Continual Learning Report\n\n{df_pivot.to_markdown(index=False)}\n\n{conclusion}")

    print(f"\n✅ Report saved to '{report_path}'")
    print(df_pivot.to_markdown(index=False))


if __name__ == "__main__":
    main()
