# ファイルパス: scripts/runners/run_hpo.py

import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_hpo.py
# Title: Optunaによるハイパーパラメータ最適化スクリプト
# Description: Optunaライブラリを使用し、指定された学習スクリプト
#              (例: run_distillation.py, train.py) のハイパーパラメータを自動調整する。
#
# 修正 (v1):
# - mypyエラー [operator] を解消するため、metric_valueがNoneでないことを確認。
# - mypyエラー [var-annotated] を解消するため、Dict, Anyをインポートし、
#   params_to_save に型ヒントを追加。
#
# 修正 (v2):
# - ユーザーの要望に基づき、サブプロセスの進捗がリアルタイムで
#   表示されるように `subprocess.run` から `subprocess.Popen` に変更。
# - ログのパースロジックを堅牢化。
#
# 修正 (v3):
# - ログに基づき、spike_reg_weight の探索範囲が大きすぎた問題を修正。
#   探索範囲を (1e-5, 1e-2) から (1e-7, 1e-4) に狭め、学習の安定化を図る。
#
# 修正 (v4):
# - Trial 64 のログに基づき、spike_reg_weight の探索範囲がまだ高すぎると判断。
#   探索範囲を (1e-10, 1e-7) にさらに狭める。
# - HPOの対象に sparsity_reg_weight を追加。
#
# 修正 (v5):
# - Trial 65 のログに基づき、spike_reg_weight と sparsity_reg_weight の
#   探索範囲がまだ高すぎると判断。探索範囲をさらに大幅に引き下げる。
#
# 修正 (v6):
# - Trial 66 のログに基づき、正則化損失が未だに発散しているため、
#   探索範囲をさらに1000倍以上引き下げる。
#
# 修正 (v7):
# - 根本原因を losses.py 側で対応（スパイク率の正規化）。
# - それに伴い、run_hpo.py の探索範囲を、直感的で妥当な
#   (例: 0.01 や 0.0001) 範囲に戻す。
#
# 修正 (v8):
# - ログ(Trial 68)の分析に基づき、v7の探索範囲ではCE/Distill損失(〜0.8)に
#   対して正則化項の寄与(〜0.0003)が小さすぎると判明。
# - losses.py(v6)で正規化された損失値(0.0-1.0)と競合できるよう、
#   重みの探索範囲を (0.1〜10.0) のオーダーに「拡大」する。

import optuna
import argparse
import subprocess
import yaml
import os
import sys
import uuid
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import json
import logging
# --- ▼ 修正: Dict, Any, List をインポート ▼ ---
from typing import Dict, Any, List
import re # 正規表現のインポートを追加
# --- ▲ 修正 ▲ ---

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトルートをPythonパスに追加 (run_hpo.pyがプロジェクトルートにあると仮定)
project_root: str = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# プロジェクト内の関数をインポート (必要に応じて)
# from app.utils import get_auto_device # 必要なら

# --- Objective Function ---
def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """Optunaの試行ごとに呼び出される目的関数。"""
    
    # --- 1. ハイパーパラメータの提案 ---
    # ここで探索したいパラメータとその範囲を定義
    # 例: 知識蒸留 (run_distillation.py) のパラメータ
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    temperature = trial.suggest_float("temperature", 1.5, 3.5)
    ce_weight = trial.suggest_float("ce_weight", 0.1, 0.5)
    distill_weight = 1.0 - ce_weight # 合わせて1になるように
    
    # --- ▼▼▼ 修正 (v8): 探索範囲を「拡大」 ▼▼▼ ---
    # losses.py (v6) でスパイク率/損失が 0.0-1.0 範囲に正規化された。
    # CE/Distill損失 (〜0.8) と競合させるため、正則化の「重み」は
    # 1e-3 のような小さな値ではなく、0.1や1.0、あるいは10.0といった
    # オーダーである必要がある。
    spike_reg_weight = trial.suggest_float("spike_reg_weight", 0.1, 10.0, log=True)
    sparsity_reg_weight = trial.suggest_float("sparsity_reg_weight", 0.01, 5.0, log=True)
    # --- ▲▲▲ 修正 (v8) ▲▲▲ ---
    
    # --- 2. 設定の上書き ---
    # 各試行にユニークな出力ディレクトリを作成
    trial_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output_base_dir) / f"trial_{trial.number}_{trial_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    overrides = [
        f"training.gradient_based.learning_rate={lr}",
        f"training.gradient_based.distillation.loss.temperature={temperature}",
        f"training.gradient_based.distillation.loss.ce_weight={ce_weight}",
        f"training.gradient_based.distillation.loss.distill_weight={distill_weight}",
        f"training.gradient_based.distillation.loss.spike_reg_weight={spike_reg_weight}",
        # --- ▼▼▼ 修正 (v4): override に sparsity_reg_weight を追加 ▼▼▼ ---
        f"training.gradient_based.distillation.loss.sparsity_reg_weight={sparsity_reg_weight}",
        # --- ▲▲▲ 修正 (v4) ▲▲▲ ---
        # 最適化中は短いエポック数で実行
        f"training.epochs={args.eval_epochs}",
        f"training.log_dir={output_dir.as_posix()}" # ログ出力先を試行ごとに変更
    ]
    
    # --- 3. 学習スクリプトの実行 ---
    command = [
        sys.executable, # 現在のPythonインタプリタを使用
        args.target_script,
        "--config", args.base_config,
        "--model_config", args.model_config,
        "--task", args.task, # run_distillation.py が task 引数を取る場合
        # 必要に応じて他の引数を追加 (例: --teacher_model)
    ]
    if args.teacher_model:
        command.extend(["--teacher_model", args.teacher_model])
        
    for override in overrides:
        command.extend(["--override_config", override])
        
    logger.info(f"--- Starting Trial {trial.number} ---")
    # --- ▼▼▼ 修正 (v4): ログに sparsity_w を追加 ▼▼▼ ---
    logger.info(f"Parameters: lr={lr:.5e}, temp={temperature:.2f}, ce_w={ce_weight:.2f}, spike_w={spike_reg_weight:.5e}, sparsity_w={sparsity_reg_weight:.5e}")
    # --- ▲▲▲ 修正 (v4) ▲▲▲ ---
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        # --- ▼▼▼ 修正 (v2): リアルタイム進捗表示 ▼▼▼ ---
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 標準エラーを標準出力にマージ
            text=True,
            encoding='utf-8',
            bufsize=1 # 行バッファリングを有効化
        )
        
        stdout_lines: List[str] = []
        
        # サブプロセスの出力をリアルタイムで読み取り、表示する
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='') # サブプロセスの進捗をそのまま表示
                stdout_lines.append(line)
        
        process.wait() # プロセスが終了するのを待つ
        
        stdout_full = "".join(stdout_lines)

        if process.returncode != 0:
            # Popenを使った場合、手動でCalledProcessErrorを発生させる
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output=stdout_full
            )
        
        logger.info(f"Trial {trial.number} finished successfully.")
        # --- ▲▲▲ 修正 (v2) ▲▲▲ ---
        
        # --- 4. 結果の解析 ---
        metric_value = float('inf') # Optunaは最小化するので損失をデフォルトに
        accuracy = 0.0 # 精度も記録しておく

        # --- ▼▼▼ 修正 (v2): ログ解析の堅牢化 ▼▼▼ ---
        # result.stdout を stdout_full に変更
        for line in reversed(stdout_full.strip().split('\n')):
            line_lower = line.lower()
            
            # train.py / run_distillation.py のログ形式 (BreakthroughTrainer.evaluate)
            # "Epoch 2 Validation Results: total: 2.5, ..., accuracy: 0.65"
            if "validation results" in line_lower and "accuracy:" in line_lower:
                try:
                    acc_str_match = re.search(r'accuracy:\s*([0-9\.]+)', line_lower)
                    acc_str = acc_str_match.group(1) if acc_str_match else "0.0"
                    accuracy = float(acc_str)
                    
                    if args.metric_name == "accuracy":
                        metric_value = -accuracy # 精度を最大化 (Optunaは最小化)
                    
                    if args.metric_name == "loss" and "total:" in line_lower:
                         loss_str_match = re.search(r'total:\s*([0-9\.]+)', line_lower)
                         loss_str = loss_str_match.group(1) if loss_str_match else "inf"
                         metric_value = float(loss_str)

                    logger.info(f"Trial {trial.number}: Found metrics from 'Validation Results': Accuracy={accuracy:.4f}, Loss={'N/A' if args.metric_name != 'loss' else metric_value:.4f}")
                    break
                except Exception as e:
                     logger.warning(f"Trial {trial.number}: Could not parse 'Validation Results' from line: '{line}'. Error: {e}")

            # run_benchmark_suite.py のログ形式 (eval_only)
            # '  - Results: {'accuracy': 0.85, ...}'
            elif "results:" in line_lower and "accuracy" in line_lower:
                 try:
                     data_str = line.split("Results:", 1)[1].strip().replace("'", "\"")
                     # テンソル表記(tensor(0.1))などを削除
                     data_str = re.sub(r'tensor\([^)]+\)', '0.0', data_str)
                     metrics = json.loads(data_str)
                     accuracy = float(metrics.get("accuracy", 0.0))
                     
                     if args.metric_name == "accuracy":
                         metric_value = -accuracy
                     elif args.metric_name == "loss":
                         metric_value = float(metrics.get("total", float('inf'))) # 'total' があれば

                     logger.info(f"Trial {trial.number}: Found metrics from 'Results:': Accuracy={accuracy:.4f}, Loss={'N/A' if args.metric_name != 'loss' else metric_value:.4f}")
                     break
                 except Exception as e:
                     logger.warning(f"Trial {trial.number}: Could not parse 'Results:' from line: '{line}'. Error: {e}")

        # --- ▲▲▲ 修正 (v2) ▲▲▲ ---
        
        # 不要になった試行ディレクトリを削除 (ディスク容量節約のためオプション)
        # shutil.rmtree(output_dir)
            
        if metric_value == float('inf') and args.metric_name == "accuracy":
             logger.warning(f"Trial {trial.number}: Could not find final 'accuracy' in log. Returning 0.0.")
             metric_value = 0.0 # 精度が見つからない場合は 0 (最大化のために -0.0)
        elif metric_value == float('inf'):
             logger.warning(f"Trial {trial.number}: Could not find final '{args.metric_name}' in log. Returning 'inf'.")
            
        return metric_value

    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed!")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return Code: {e.returncode}")
        # --- ▼ 修正 (v2): e.stdout/stderr を e.output に変更 ▼ ---
        logger.error(f"Output:\n{e.output}")
        # --- ▲ 修正 (v2) ▲ ---
        return float('inf') # 最小化の場合
    except Exception as e:
        logger.error(f"An unexpected error occurred in trial {trial.number}: {e}", exc_info=True)
        return float('inf')

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization using Optuna")
    parser.add_argument("--target_script", type=str, default="run_distillation.py", help="学習を実行するターゲットスクリプト (例: run_distillation.py, train.py)")
    parser.add_argument("--base_config", type=str, default="configs/templates/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, required=True, help="モデルアーキテクチャ設定ファイル (例: configs/experiments/cifar10_spikingcnn_config.yaml)")
    parser.add_argument("--task", type=str, required=True, help="ターゲットタスク (例: cifar10)")
    parser.add_argument("--teacher_model", type=str, help="教師モデル (run_distillation.py用)")
    parser.add_argument("--n_trials", type=int, default=50, help="Optunaの試行回数")
    parser.add_argument("--eval_epochs", type=int, default=3, help="各試行で実行するエポック数")
    parser.add_argument("--metric_name", type=str, default="accuracy", choices=["accuracy", "loss"], help="最適化するメトリクス ('accuracy' または 'loss')")
    parser.add_argument("--output_base_dir", type=str, default="runs/hpo_trials", help="各試行のログを保存するベースディレクトリ")
    parser.add_argument("--study_name", type=str, default="snn_hpo_study", help="Optuna Studyの名前 (DB保存用)")
    parser.add_argument("--storage", type=str, default="sqlite:///runs/hpo_study.db", help="OptunaのDB保存先 (例: sqlite:///study.db)")
    
    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    storage_path_dir = os.path.dirname(args.storage.replace("sqlite:///", ""))
    if storage_path_dir: # storageがファイルパスの場合のみ
        Path(storage_path_dir).mkdir(parents=True, exist_ok=True)

    # Optuna Studyの作成と最適化の実行
    direction = "maximize" if args.metric_name == "accuracy" else "minimize"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True, # 既存の研究があれば再開
        direction=direction
    )
    
    logger.info(f"Starting Optuna optimization for {args.n_trials} trials...")
    logger.info(f"Optimizing '{args.metric_name}' ({direction})")
    logger.info(f"Study Name: {args.study_name}, Storage: {args.storage}")
    
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=None) # Timeoutなし
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
        
    # 最適化結果の表示
    logger.info("--- Optimization Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    try:
        best_trial = study.best_trial
        metric_value = best_trial.value
        # --- ▼ 修正: mypy [operator] ▼ ---
        if args.metric_name == "accuracy" and metric_value is not None:
            metric_value = -metric_value # 精度に戻す
        # --- ▲ 修正 ▲ ---
            
        logger.info(f"Best trial (Trial {best_trial.number}):")
        logger.info(f"  Value ({args.metric_name}): {metric_value:.4f}")
        logger.info("  Best Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        # 最適パラメータをYAMLファイルとして保存
        best_params_yaml = Path(args.output_base_dir) / "best_params.yaml"
        with open(best_params_yaml, 'w') as f:
            # パラメータをOmegaConfが読み込める形式で保存
            # --- ▼ 修正: mypy [var-annotated] ▼ ---
            params_to_save: Dict[str, Any] = {}
            # --- ▲ 修正 ▲ ---
            for key, value in best_trial.params.items():
                # '.'区切りでネスト構造を作成
                keys = key.split('.')
                d = params_to_save
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            yaml.dump(params_to_save, f, default_flow_style=False)
        logger.info(f"Best parameters saved to: {best_params_yaml}")
        
    except ValueError:
         logger.warning("No completed trials found in the study. Could not determine best parameters.")