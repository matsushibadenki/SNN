# ファイルパス: scripts/runners/run_hpo.py
# (修正: os モジュールのインポート漏れを修正)

import optuna
import argparse
import subprocess
import sys
import uuid
from pathlib import Path
import logging
from typing import Dict, Any, List
import re
import json
import yaml
# --- ▼ 追加 ▼ ---
import os
# --- ▲ 追加 ▲ ---

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Objective Function ---
def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """Optunaの試行ごとに呼び出される目的関数。"""
    
    # --- 1. ハイパーパラメータの提案 ---
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    temperature = trial.suggest_float("temperature", 1.5, 3.5)
    ce_weight = trial.suggest_float("ce_weight", 0.1, 0.5)
    distill_weight = 1.0 - ce_weight 
    
    spike_reg_weight = trial.suggest_float("spike_reg_weight", 0.1, 10.0, log=True)
    sparsity_reg_weight = trial.suggest_float("sparsity_reg_weight", 0.01, 5.0, log=True)
    
    # --- 2. 設定の上書き ---
    trial_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output_base_dir) / f"trial_{trial.number}_{trial_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    overrides = [
        f"training.gradient_based.learning_rate={lr}",
        f"training.gradient_based.distillation.loss.temperature={temperature}",
        f"training.gradient_based.distillation.loss.ce_weight={ce_weight}",
        f"training.gradient_based.distillation.loss.distill_weight={distill_weight}",
        f"training.gradient_based.distillation.loss.spike_reg_weight={spike_reg_weight}",
        f"training.gradient_based.distillation.loss.sparsity_reg_weight={sparsity_reg_weight}",
        f"training.epochs={args.eval_epochs}",
        f"training.log_dir={output_dir.as_posix()}"
    ]
    
    # --- 3. 学習スクリプトの実行 ---
    command = [
        sys.executable,
        args.target_script,
        "--config", args.base_config,
        "--model_config", args.model_config,
    ]
    
    if args.task:
         if "train.py" in args.target_script:
             command.extend(["--task_name", args.task])
         else:
             command.extend(["--task", args.task])

    if args.teacher_model:
        command.extend(["--teacher_model", args.teacher_model]) 
        command.extend(["--override_config", f"training.gradient_based.distillation.teacher_model={args.teacher_model}"])
        
    for override in overrides:
        command.extend(["--override_config", override])
        
    logger.info(f"--- Starting Trial {trial.number} ---")
    logger.info(f"Parameters: lr={lr:.5e}, temp={temperature:.2f}, ce_w={ce_weight:.2f}, spike_w={spike_reg_weight:.5e}, sparsity_w={sparsity_reg_weight:.5e}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        stdout_lines: List[str] = []
        
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                stdout_lines.append(line)
        
        process.wait()
        
        stdout_full = "".join(stdout_lines)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output=stdout_full
            )
        
        logger.info(f"Trial {trial.number} finished successfully.")
        
        # --- 4. 結果の解析 ---
        metric_value = float('inf')
        accuracy = 0.0

        for line in reversed(stdout_full.strip().split('\n')):
            line_lower = line.lower()
            
            # train.py / run_distillation.py のログ形式
            if "validation results" in line_lower and "accuracy:" in line_lower:
                try:
                    acc_str_match = re.search(r'accuracy:\s*([0-9\.]+)', line_lower)
                    acc_str = acc_str_match.group(1) if acc_str_match else "0.0"
                    accuracy = float(acc_str)
                    
                    if args.metric_name == "accuracy":
                        metric_value = -accuracy
                    
                    if args.metric_name == "loss" and "total:" in line_lower:
                         loss_str_match = re.search(r'total:\s*([0-9\.]+)', line_lower)
                         loss_str = loss_str_match.group(1) if loss_str_match else "inf"
                         metric_value = float(loss_str)

                    logger.info(f"Trial {trial.number}: Found metrics from 'Validation Results': Accuracy={accuracy:.4f}, Loss={'N/A' if args.metric_name != 'loss' else metric_value:.4f}")
                    break
                except Exception as e:
                     logger.warning(f"Trial {trial.number}: Could not parse 'Validation Results' from line: '{line}'. Error: {e}")

            # run_benchmark_suite.py のログ形式
            elif "results:" in line_lower and "accuracy" in line_lower:
                 try:
                     data_str = line.split("Results:", 1)[1].strip().replace("'", "\"")
                     data_str = re.sub(r'tensor\([^)]+\)', '0.0', data_str)
                     metrics = json.loads(data_str)
                     accuracy = float(metrics.get("accuracy", 0.0))
                     
                     if args.metric_name == "accuracy":
                         metric_value = -accuracy
                     elif args.metric_name == "loss":
                         metric_value = float(metrics.get("total", float('inf')))

                     logger.info(f"Trial {trial.number}: Found metrics from 'Results:': Accuracy={accuracy:.4f}, Loss={'N/A' if args.metric_name != 'loss' else metric_value:.4f}")
                     break
                 except Exception as e:
                     logger.warning(f"Trial {trial.number}: Could not parse 'Results:' from line: '{line}'. Error: {e}")

        
        if metric_value == float('inf') and args.metric_name == "accuracy":
             logger.warning(f"Trial {trial.number}: Could not find final 'accuracy' in log. Returning 0.0.")
             metric_value = 0.0
        elif metric_value == float('inf'):
             logger.warning(f"Trial {trial.number}: Could not find final '{args.metric_name}' in log. Returning 'inf'.")
            
        return metric_value

    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed!")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"Output:\n{e.output}")
        return float('inf')
    except Exception as e:
        logger.error(f"An unexpected error occurred in trial {trial.number}: {e}", exc_info=True)
        return float('inf')

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization using Optuna")
    parser.add_argument("--target_script", type=str, default="scripts/runners/train.py", help="学習スクリプト")
    parser.add_argument("--base_config", type=str, default="configs/templates/base_config.yaml")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--teacher_model", type=str)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--eval_epochs", type=int, default=3)
    parser.add_argument("--metric_name", type=str, default="accuracy")
    parser.add_argument("--output_base_dir", type=str, default="workspace/runs/hpo_trials")
    parser.add_argument("--study_name", type=str, default="snn_hpo_study")
    parser.add_argument("--storage", type=str, default="sqlite:///runs/hpo_study.db")
    
    args = parser.parse_args()

    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    storage_path_dir = os.path.dirname(args.storage.replace("sqlite:///", ""))
    if storage_path_dir:
        Path(storage_path_dir).mkdir(parents=True, exist_ok=True)

    direction = "maximize" if args.metric_name == "accuracy" else "minimize"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=direction
    )
    
    logger.info(f"Starting Optuna optimization for {args.n_trials} trials...")
    logger.info(f"Optimizing '{args.metric_name}' ({direction})")
    
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=None)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
        
    logger.info("--- Optimization Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    try:
        best_trial = study.best_trial
        metric_value = best_trial.value
        if args.metric_name == "accuracy" and metric_value is not None:
            metric_value = -metric_value
            
        logger.info(f"Best trial (Trial {best_trial.number}):")
        logger.info(f"  Value ({args.metric_name}): {metric_value:.4f}")
        logger.info("  Best Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        best_params_yaml = Path(args.output_base_dir) / "best_params.yaml"
        with open(best_params_yaml, 'w') as f:
            params_to_save: Dict[str, Any] = {}
            for key, value in best_trial.params.items():
                keys = key.split('.')
                d = params_to_save
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            yaml.dump(params_to_save, f, default_flow_style=False)
        logger.info(f"Best parameters saved to: {best_params_yaml}")
        
    except ValueError:
         logger.warning("No completed trials found in the study. Could not determine best parameters.")