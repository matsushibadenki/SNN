#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ファイルパス: snn-cli.py
タイトル: SNN Project CLI Tool
機能説明:
SNNプロジェクトの主要なワークフロー（学習、評価、変換、UI起動、最適化、診断など）を
統一的に管理するためのコマンドラインインターフェース（CLI）ツール。
clickライブラリを使用して、モジュール化されたコマンドグループを提供します。
"""

import click
import os
import sys
import subprocess
import logging
from glob import glob
import shutil

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# --- ヘルパー関数 ---

def find_python_executable():
    """実行中のPythonインタプリタのパスを返す"""
    return sys.executable

def run_script(script_path, args, capture_output=False):
    """
    指定されたPythonスクリプトをサブプロセスとして実行する。
    
    Args:
        script_path (str): 実行するスクリプトのパス。
        args (list): スクリプトに渡す引数のリスト。
        capture_output (bool): 標準出力をキャプチャするかどうか。
    
    Returns:
        subprocess.CompletedProcess: 実行結果。
    """
    python_exec = find_python_executable()
    command = [python_exec, script_path] + args
    
    logger.info(f"実行中: {' '.join(command)}")
    
    try:
        if capture_output:
            # capture_output=True は Python 3.7+
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(command, check=True, text=True)
            
        logger.info(f"スクリプト {script_path} が正常に完了しました。")
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"スクリプト実行中にエラーが発生しました: {script_path}")
        logger.error(f"コマンド: {' '.join(command)}")
        logger.error(f"リターンコード: {e.returncode}")
        if capture_output:
            logger.error(f"標準出力: {e.stdout}")
            logger.error(f"標準エラー: {e.stderr}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"スクリプトが見つかりません: {script_path}")
        logger.error("プロジェクトのルートディレクトリから実行しているか確認してください。")
        sys.exit(1)

def run_external_command(command_list, capture_output=False):
    """
    指定された外部コマンドをサブプロセスとして実行する。
    Lintツール（flake8, mypy）など、エラーコードを返す可能性があるが
    処理を続行したい場合に使用します（check=False）。
    
    Args:
        command_list (list): 実行するコマンドと引数のリスト。
        capture_output (bool): 標準出力をキャプチャするかどうか。
    
    Returns:
        subprocess.CompletedProcess: 実行結果。
    """
    logger.info(f"実行中: {' '.join(command_list)}")
    
    try:
        if capture_output:
            result = subprocess.run(command_list, check=False, text=True, capture_output=True)
        else:
            # check=False (デフォルト) で実行し、LintエラーでCLIが停止しないようにする
            result = subprocess.run(command_list, text=True)
            
        if result.returncode == 0:
            logger.info(f"コマンド {command_list[0]} が正常に完了しました。")
        else:
            logger.warning(f"コマンド {command_list[0]} が完了しました (リターンコード: {result.returncode})。Lintエラー/警告の可能性があります。")
            
        return result
        
    except FileNotFoundError:
        logger.error(f"コマンドが見つかりません: {command_list[0]}")
        logger.error(f"{command_list[0]} がインストールされているか、PATHが通っているか確認してください。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"コマンド {command_list[0]} 実行中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. メインCLIグループ ---
@click.group()
def cli():
    """SNNプロジェクト 統合CLIツール (v11.0)"""
    pass

# --- 2. クリーンアップ機能 (v10.9で空ディレクトリ削除機能を追加) ---
@click.group()
def clean_cli():
    """プロジェクトのクリーンアップ (ログ、キャッシュ、モデル、データ)"""
    pass

def _perform_clean(delete_models=False, delete_data=False, yes=False):
    """クリーンアップ処理の共通ロジック"""
    # 常にクリーンアップ対象とするディレクトリ
    targets = ['runs', 'workspace', 'results']
    
    # データ削除モードの時のみ対象とするディレクトリ (事前計算ロジットなど)
    # これにより、clean models で .pt ファイルが誤って消されるのを防ぐ
    if delete_data:
        targets.append('precomputed_data')
    
    # 保護対象の基本セット (ドキュメントは常に保護)
    protected_extensions = {'.md'}
    
    # データ保護モード (delete_data=False) の場合、データファイルを保護
    if not delete_data:
        protected_extensions.update({'.jsonl', '.json', '.db', '.csv'})
        
    # モデル保護モード (delete_models=False) の場合、モデルファイルを保護
    if not delete_models:
        protected_extensions.update({'.pth', '.pt', '.safetensors'})

    files_to_delete = []
    
    # 1. ファイルの削除リスト作成
    for target_dir in targets:
        if os.path.isdir(target_dir):
            for filepath in glob(os.path.join(target_dir, '**', '*'), recursive=True):
                if os.path.isfile(filepath):
                    # ファイルの拡張子が保護対象セットのいずれにも一致しない場合、削除対象
                    file_ext = os.path.splitext(filepath)[1]
                    if file_ext not in protected_extensions:
                        files_to_delete.append(filepath)

    if not files_to_delete:
        logger.info("削除対象のファイルが見つかりません。")
    else:
        logger.info("以下のファイルが削除されます:")
        for f in files_to_delete[:20]:
            logger.info(f"  {f}")
        if len(files_to_delete) > 20:
            logger.info(f"  ...他 {len(files_to_delete) - 20} ファイル")

        if not yes:
            click.confirm(f"{len(files_to_delete)}個のファイルを削除してもよろしいですか？", abort=True)
        
        deleted_count = 0
        for f in files_to_delete:
            try:
                os.remove(f)
                deleted_count += 1
            except OSError as e:
                logger.warning(f"ファイルを削除できませんでした: {f} (エラー: {e})")
                
        logger.info(f"{deleted_count} / {len(files_to_delete)} 個のファイルを削除しました。")

    # 2. 空ディレクトリのクリーンアップ
    logger.info("空になったディレクトリを整理しています...")
    removed_dirs_count = 0
    
    for target_dir in targets:
        if os.path.isdir(target_dir):
            # bottomdown=True (topdown=False) で最深部から探索して削除
            for root, dirs, files in os.walk(target_dir, topdown=False):
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        # ディレクトリが空かどうか確認（.DS_Storeなどは無視して削除したい場合はここで処理が必要だが、基本は空のみ）
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            removed_dirs_count += 1
                    except OSError:
                        pass # 削除に失敗した場合（権限やファイルが残っている場合）は無視
    
    if removed_dirs_count > 0:
        logger.info(f"{removed_dirs_count} 個の空ディレクトリを削除しました。")

@clean_cli.command(name="logs")
@click.option('--yes', '-y', is_flag=True, help="確認プロンプトをスキップします。")
def clean_logs(yes):
    """
    ログとキャッシュのみを削除します。
    モデルファイル (.pth, .pt) やデータファイル (.jsonl, .csv) は保護されます。
    """
    logger.info("モード: ログとキャッシュのみをクリーンアップします。")
    _perform_clean(delete_models=False, delete_data=False, yes=yes)

@clean_cli.command(name="models")
@click.option('--yes', '-y', is_flag=True, help="確認プロンプトをスキップします。")
def clean_models(yes):
    """
    ログ、キャッシュ、および学習済みモデル (.pth, .pt) を削除します。
    データファイル (.jsonl, .csv) および事前計算データ (precomputed_data) は保護されます。
    """
    logger.info("モード: ログ、キャッシュ、モデルをクリーンアップします。")
    _perform_clean(delete_models=True, delete_data=False, yes=yes)

@clean_cli.command(name="all")
@click.option('--yes', '-y', is_flag=True, help="確認プロンプトをスキップします。")
def clean_all(yes):
    """
    すべての生成物（ログ、キャッシュ、モデル、データ）を削除します。
    （注意：precomputed_data内の.jsonlなども削除されます）
    """
    logger.info("モード: すべての生成物をクリーンアップします。")
    _perform_clean(delete_models=True, delete_data=True, yes=yes)


cli.add_command(clean_cli, name="clean")

# --- 3. ハイパーパラメータ最適化 (HPO) / チューニング (Tune) ---
@click.group()
def hpo_cli():
    """ハイパーパラメータ最適化 (Optuna) / チューニング"""
    pass

@hpo_cli.command(name="run")
@click.argument('model_config', type=click.Path(exists=True))
@click.argument('task_name', type=str)
@click.option('--target-script', default="run_distillation.py", help="最適化対象のスクリプト。")
@click.option('--teacher-model', default="models/ann_teacher.pth", help="教師モデルのパス。")
@click.option('--n-trials', default=50, help="試行回数。")
@click.option('--eval-epochs', default=5, help="各試行での評価エポック数。")
@click.option('--metric-name', default="accuracy", help="最適化するメトリクス名。")
@click.option('--storage', default=None, help="OptunaのストレージURL (例: sqlite:///db.sqlite3)。")
@click.option('--study-name', default=None, help="Optunaのスタディ名。")
def hpo_run(model_config, task_name, target_script, teacher_model, n_trials, eval_epochs, metric_name, storage, study_name):
    """
    run_hpo.py を実行してハイパーパラメータ最適化を開始します。
    
    MODEL_CONFIG: モデル設定ファイル (例: configs/models/medium.yaml)\n
    TASK_NAME: タスク名 (例: cifar10, sst2)
    """
    script_path = "run_hpo.py"
    args = [
        "--model_config", model_config,
        "--task", task_name,
        "--target-script", target_script,
        "--teacher-model", teacher_model,
        "--n-trials", str(n_trials),
        "--eval-epochs", str(eval_epochs),
        "--metric-name", metric_name,
    ]
    if storage:
        args.extend(["--storage", storage])
    if study_name:
        args.extend(["--study_name", study_name])

    run_script(script_path, args)

cli.add_command(hpo_cli, name="hpo")
cli.add_command(hpo_cli, name="tune") # 'tune' をエイリアスとして追加


# --- 3.5. 自動チューニング ---
@hpo_cli.command(name="efficiency")
@click.option('--n-trials', default=20, help="最適化の試行回数。")
@click.option('--config', default="configs/experiments/smoke_test_config.yaml", help="ベースとなる学習設定。")
@click.option('--model-config', default="configs/models/micro.yaml", help="最適化するモデル設定。")
@click.option('--data-path', default="data/smoke_test_data.jsonl", help="使用するデータセット。")
def hpo_efficiency(n_trials, config, model_config, data_path):
    """
    スパイク率(<7%)と精度のバランスを最適化する自動チューニングを実行します。
    (scripts/auto_tune_efficiency.py を実行)
    """
    script_path = "scripts/auto_tune_efficiency.py"
    args = [
        "--n-trials", str(n_trials),
        "--config", config,
        "--model-config", model_config,
        "--data-path", data_path
    ]
    run_script(script_path, args)
    
    
# --- 4. UI起動 ---
@click.group()
def ui_cli():
    """Gradio UIの起動"""
    pass

@ui_cli.command(name="start")
@click.option('--start-langchain', is_flag=True, help="app/langchain_main.py を起動します。")
@click.option('--model-config', default="configs/models/small.yaml", help="（標準UI用）モデル設定ファイル。")
@click.option('--model-path', type=click.Path(exists=True), default=None, help="（標準UI用）学習済みモデルパス。")
# (動的ロード用オプション)
@click.option('--cifar_model_config', help="（動的ロード用）CIFARモデル設定")
def ui_start(start_langchain, model_config, model_path, **kwargs):
    """
    Gradio UIアプリケーションを起動します。
    
    デフォルトは app/main.py を起動します。
    --start-langchain フラグで app/langchain_main.py を起動します。
    """
    if start_langchain:
        script_path = "app/langchain_main.py"
        logger.info("LangChain連携UIを起動します...")
    else:
        script_path = "app/main.py"
        logger.info("標準UIを起動します...")

    args = [
        "--model_config", model_config
    ]
    if model_path:
        args.extend(["--model_path", model_path])

    # 動的ロード用の引数を追加
    for key, value in kwargs.items():
        if value:
            # key をスネークケースから引数名（例: --ai_tech_model_path）に変換
            arg_name = f"--{key}"
            args.extend([arg_name, value])

    run_script(script_path, args)

cli.add_command(ui_cli, name="ui")


# --- 5. モデル学習 (Train) ---
@click.group()
def train_cli():
    """SNNモデルの学習"""
    pass

@train_cli.command(name="gradient")
@click.option('--config', default="configs/experiments/cifar10_spikingcnn_config.yaml", help="学習設定ファイル。")
@click.option('--model-config', required=True, help="モデル設定ファイル。")
@click.option('--data-path', default="data/", help="データセットのパス。")
@click.option('--override-config', default=None, help="設定の一部を上書き (例: 'optimizer.lr=0.01')。")
@click.option('--resume-path', default=None, type=click.Path(exists=True), help="学習を再開するチェックポイント。")
@click.option('--distributed', is_flag=True, help="分散学習を有効にする。")
@click.option('--task-name', default=None, help="タスク名（EWCなどで使用）。")
@click.option('--load-ewc-data', default=None, help="事前計算されたEWCデータのパス。")
def gradient_train(config, model_config, data_path, override_config, resume_path, distributed, task_name, load_ewc_data):
    """
    train.py を実行し、代理勾配法(SG)によるSNNの直接学習を行います。
    """
    script_path = "train.py"
    args = [
        "--config", config,
        "--model_config", model_config,
        "--data_path", data_path,
    ]
    if override_config:
        args.extend(["--override_config", override_config])
    if resume_path:
        args.extend(["--resume_path", resume_path])
    if distributed:
        args.append("--distributed")
    if task_name:
        args.extend(["--task_name", task_name])
    if load_ewc_data:
        args.extend(["--load_ewc_data", load_ewc_data])

    run_script(script_path, args)

@train_cli.command(name="distill")
@click.option('--task', required=True, help="タスク名 (例: cifar10, sst2)。")
@click.option('--teacher-model', required=True, help="教師モデルのパス (例: models/ann_teacher.pth)。")
@click.option('--model-config', required=True, help="学習させるSNN生徒モデルの設定ファイル。")
@click.option('--epochs', default=15, help="学習エポック数。")
def train_distill(task, teacher_model, model_config, epochs):
    """
    run_distillation.py を実行し、知識蒸留を行います。
    """
    script_path = "run_distillation.py"
    args = [
        "--task", task,
        "--teacher_model", teacher_model,
        "--model-config", model_config,
        "--epochs", str(epochs),
    ]
    run_script(script_path, args)

cli.add_command(train_cli, name="train")


# --- 6. ベンチマーク & 評価 (Benchmark & Evaluation) ---
@click.group()
def benchmark_cli():
    """モデルの評価とベンチマーク"""
    pass

@benchmark_cli.command(name="run")
@click.option('--experiment', required=True, help="実行する実験設定 (例: cifar10_ann_vs_snn)。")
@click.option('--epochs', default=5, help="学習エポック数。")
@click.option('--tag', default="BenchmarkRun", help="実験タグ。")
def benchmark_run(experiment, epochs, tag):
    """
    run_benchmark_suite.py を実行し、SNNとANNの性能比較を行います (学習＋評価)。
    """
    script_path = "scripts/run_benchmark_suite.py"
    args = [
        "--experiment", experiment,
        "--epochs", str(epochs),
        "--tag", tag,
    ]
    run_script(script_path, args)

@benchmark_cli.command(name="evaluate-accuracy")
@click.option('--model-path', required=True, type=click.Path(exists=True), help="評価する学習済みモデルのパス (.pth)。")
@click.option('--model-config', required=True, help="モデル設定ファイル。")
@click.option('--model-type', required=True, type=click.Choice(['snn', 'ann']), help="モデルタイプ (snn または ann)。")
@click.option('--experiment', required=True, help="実行する実験設定 (例: cifar10)。")
@click.option('--tag', default="AccuracyEvaluation", help="評価実行のタグ。")
def evaluate_accuracy(model_path, model_config, model_type, experiment, tag):
    """
    学習済みモデルの「精度」を評価します (評価専用モード)。
    """
    script_path = "scripts/run_benchmark_suite.py"
    args = [
        "--eval_only",
        "--model_path", model_path,
        "--model_config", model_config,
        "--model_type", model_type,
        "--experiment", experiment,
        "--tag", tag,
    ]
    run_script(script_path, args)

@benchmark_cli.command(name="continual")
@click.option('--epochs-task-a', default=3, help="タスクAの学習エポック数。")
@click.option('--epochs-task-b', default=3, help="タスクBの学習エポック数。")
def benchmark_continual(epochs_task_a, epochs_task_b):
    """
    run_continual_learning_experiment.py を実行し、継続学習の性能を評価します。
    """
    script_path = "scripts/run_continual_learning_experiment.py"
    args = [
        "--epochs_task_a", str(epochs_task_a),
        "--epochs_task_b", str(epochs_task_b),
    ]
    run_script(script_path, args)

cli.add_command(benchmark_cli, name="benchmark")


# --- 7. モデル変換 (Conversion) ---
@click.group()
def convert_cli():
    """モデルの変換 (ANN -> SNN)"""
    pass

@convert_cli.command(name="ann2snn-cnn")
@click.argument('ann_model_path', type=click.Path(exists=True))
@click.argument('output_snn_path', type=click.Path())
@click.option('--snn-model-config', required=True, help="変換後のSNNモデルの設定ファイル。")
def convert_ann2snn_cnn(ann_model_path, output_snn_path, snn_model_config):
    """
    convert_model.py を実行し、学習済みANN (CNN) を SNN に変換します。
    
    ANN_MODEL_PATH: 変換元のANNモデルパス (.pth)\n
    OUTPUT_SNN_PATH: 変換後のSNNモデル出力パス (.pth)
    """
    script_path = "scripts/convert_model.py"
    args = [
        "--ann_model_path", ann_model_path, # 引数名を明示
        "--output_snn_path", output_snn_path, # 引数名を明示
        "--method", "cnn-convert", # メソッドを明示
        "--snn_model_config", snn_model_config,
    ]
    run_script(script_path, args)

cli.add_command(convert_cli, name="convert")

# --- 8. エージェント・認知システム (Agents) ---
@click.group()
def agent_cli():
    """自律エージェントと認知アーキテクチャ"""
    pass

@agent_cli.command(name="solve")
@click.option('--task-description', required=True, help="解決すべきタスクの自然言語記述。")
@click.option('--unlabeled-data-path', default=None, help="（オプション）自己学習に使用する未ラベルデータ。")
@click.option('--force-retrain', is_flag=True, help="既存モデルを無視して強制的に再学習。")
def agent_solve(task_description, unlabeled_data_path, force_retrain):
    """run_agent.py を実行し、タスク解決エージェントを起動。"""
    script_path = "run_agent.py"
    args = ["--task_description", task_description]
    if unlabeled_data_path:
        args.extend(["--unlabeled_data_path", unlabeled_data_path])
    if force_retrain:
        args.append("--force_retrain")
    run_script(script_path, args)

@agent_cli.command(name="evolve")
@click.option('--task-description', required=True, help="進化の指針となるタスク記述。")
def agent_evolve(task_description):
    """run_evolution.py を実行し、自己進化エージェントを起動。"""
    script_path = "run_evolution.py"
    args = ["--task_description", task_description]
    run_script(script_path, args)

@agent_cli.command(name="rl")
@click.option('--episodes', default=1000, help="実行するエピソード数。")
def agent_rl(episodes):
    """run_rl_agent.py を実行し、強化学習エージェントを起動。"""
    script_path = "run_rl_agent.py"
    args = ["--episodes", str(episodes)]
    run_script(script_path, args)

@agent_cli.command(name="planner")
@click.option('--task-request', required=True, help="実行すべきタスク要求。")
@click.option('--context-data', required=True, help="タスクの文脈データ。")
def agent_planner(task_request, context_data):
    """run_planner.py を実行し、階層的プランナーを起動。"""
    script_path = "run_planner.py"
    args = ["--task_request", task_request, "--context_data", context_data]
    run_script(script_path, args)

@agent_cli.command(name="brain")
@click.option('--prompt', default="今日の天気は？", help="人工脳への初期入力。")
@click.option('--loop', is_flag=True, help="対話ループ (observe_brain_thought_process.py) を実行。")
@click.option('--model-config', default='configs/models/small.yaml', help="人工脳が使用するモデル設定。")
def agent_brain(prompt, loop, model_config):
    """
    人工脳シミュレーション (run_brain_simulation.py または observe_brain_thought_process.py) を実行。
    """
    args = ["--model_config", model_config]
    if loop:
        script_path = "scripts/observe_brain_thought_process.py"
    else:
        script_path = "run_brain_simulation.py"
        args.extend(["--prompt", prompt])
    run_script(script_path, args)

@agent_cli.command(name="life-form")
@click.option('--duration', default=60, help="デジタル生命体の実行時間（秒）。")
@click.option('--model-config', default='configs/models/small.yaml', help="生命体が使用するモデル設定。")
def agent_life_form(duration, model_config):
    """run_life_form.py を実行し、デジタル生命体シミュレーションを開始。"""
    script_path = "run_life_form.py"
    args = ["--duration", str(duration), "--model_config", model_config]
    run_script(script_path, args)

cli.add_command(agent_cli, name="agent")


# --- 9. 診断 (Diagnostics) ---
@click.group()
def diagnostics_cli():
    """モデルの診断 (スパース性、レイテンシ)"""
    pass

@diagnostics_cli.command(name="report-efficiency")
@click.option('--model-config', required=True, help="診断するモデル設定ファイル。")
@click.option('--data-path', default="data/cifar10", help="診断に使用するデータパス。")
@click.option('--model-path', type=click.Path(exists=True), help="(オプション) 学習済みモデルのパス (.pth)。")
def report_efficiency(model_config, data_path, model_path):
    """
    学習済みSNNモデルの「効率」（スパース性・タイムステップ）を診断します。
    (scripts/report_sparsity_and_T.py を実行)
    """
    script_path = "scripts/report_sparsity_and_T.py"
    args = [
        "--model_config", model_config,
        "--data_path", data_path,
    ]
    if model_path:
        args.extend(["--model_path", model_path])
    run_script(script_path, args)

cli.add_command(diagnostics_cli, name="diagnostics")

# --- 10. ヘルスチェック ---
@cli.command(name="health-check")
def health_check():
    """
    プロジェクトの主要機能（学習、ベンチマーク、エージェント等）の
    健全性チェック (scripts/run_project_health_check.py) を実行します。
    """
    script_path = "scripts/run_project_health_check.py"
    args = []
    run_script(script_path, args)

# --- 11. データ管理 (Data) ---
@click.group()
def data_cli():
    """データセットの準備、構築、管理"""
    pass

@data_cli.command(name="prepare")
@click.option('--dataset', required=True, help="準備するデータセット名 (例: cifar10, sst2)。")
@click.option('--output-dir', default='data/prepared', help="前処理済みデータの出力先。")
def data_prepare(dataset, output_dir):
    """
    データの前処理（ダウンロード、変換）を実行します。
    (scripts/data_preparation.py を実行)
    """
    script_path = "scripts/data_preparation.py"
    args = [
        "--dataset", dataset,
        "--output_dir", output_dir
    ]
    run_script(script_path, args)

@data_cli.command(name="build-kb")
@click.option('--input-dir', default='data/knowledge_source', help="知識ベースの元データディレクトリ。")
@click.option('--output-db', default='workspace/knowledge_base.db', help="構築するVectorDBのパス。")
def data_build_kb(input_dir, output_db):
    """
    RAGエージェント用の知識ベース（VectorDB）を構築します。
    (scripts/build_knowledge_base.py を実行)
    """
    script_path = "scripts/build_knowledge_base.py"
    args = [
        "--input_dir", input_dir,
        "--output_db", output_db
    ]
    run_script(script_path, args)

@data_cli.command(name="prep-distill")
@click.option('--task', required=True, help="タスク名 (例: cifar10, sst2)。")
@click.option('--teacher-model', required=True, help="教師モデルのパス (例: models/ann_teacher.pth)。")
@click.option('--output-dir', default='precomputed_data/distillation', help="教師モデルの出力（ロジット）の保存先。")
def data_prep_distill(task, teacher_model, output_dir):
    """
    知識蒸留用の教師モデルの出力を事前計算します。
    (scripts/prepare_distillation_data.py を実行)
    """
    script_path = "scripts/prepare_distillation_data.py"
    args = [
        "--task", task,
        "--teacher_model", teacher_model,
        "--output_dir", output_dir
    ]
    run_script(script_path, args)

cli.add_command(data_cli, name="data")

# --- 12. デバッグ (Debug) ---
@click.group()
def debug_cli():
    """プロジェクトのデバッグとコード分析"""
    pass

@debug_cli.command(name="analyze")
@click.option('--tool', default='all', type=click.Choice(['all', 'flake8', 'mypy']), help="実行する分析ツール。")
@click.option('--skip-mypy', is_flag=True, help="mypy（型チェック）をスキップします。")
@click.option('--skip-flake8', is_flag=True, help="flake8（スタイル/Lint）をスキップします。")
def debug_analyze(tool, skip_mypy, skip_flake8):
    """
    プロジェクト全体のコードを静的解析ツール (flake8, mypy) で調査します。
    
    これらのツールが環境にインストールされている必要があります。
    (例: pip install flake8 mypy)
    """
    targets = ["snn_research", "app", "scripts", "tests"]
    # ルートの .py ファイルを追加
    targets.extend(glob("*.py"))
    
    run_flake8 = (tool in ['all', 'flake8']) and not skip_flake8
    run_mypy = (tool in ['all', 'mypy']) and not skip_mypy

    if not run_flake8 and not run_mypy:
        logger.warning("実行する分析ツールがありません。")
        return

    if run_flake8:
        logger.info("--- flake8 (スタイル/Lintチェック) を実行します ---")
        flake8_command = ["flake8"] + targets
        run_external_command(flake8_command)
        logger.info("--- flake8 完了 ---")

    if run_mypy:
        logger.info("--- mypy (型チェック) を実行します ---")
        # mypyは .py ファイルを直接渡すとエラーになることがあるため、ディレクトリと主要ファイルのみ
        mypy_targets = ["snn_research", "app", "scripts", "tests", "snn-cli.py", "train.py"]
        mypy_command = ["mypy"] + mypy_targets
        run_external_command(mypy_command)
        logger.info("--- mypy 完了 ---")

@debug_cli.command(name="spike-test")
@click.option('--model-config', required=True, type=click.Path(exists=True), 
              help="テストするSNNモデルのアーキテクチャ設定 (例: configs/models/micro.yaml)。")
@click.option('--timesteps', default=16, help="テストに使用するタイムステップ数。")
@click.option('--batch-size', default=4, help="テストに使用するバッチサイズ。")
def debug_spike_test(model_config, timesteps, batch_size):
    """
    SNNモデルの基本的なスパイク活動をテストします (学習重みなし)。
    (scripts/debug_spike_activity.py を実行)
    
    モデルがランダム入力に対してスパイクを生成するか、発火率が異常でないか（例：0%や100%でないか）
    を確認するための、低レベルな健全性チェックです。
    """
    script_path = "scripts/debug_spike_activity.py"
    args = [
        "--model_config", model_config,
        "--timesteps", str(timesteps),
        "--batch_size", str(batch_size),
    ]
    run_script(script_path, args)

@debug_cli.command(
    "spike-visualize",
    help="スパイク活動（ラスタプロット）を詳細に可視化し、分析します。",
)
@click.option(
    "--model-config",
    required=True,
    type=click.Path(exists=True),
    help="モデル設定ファイル (.yaml)。",
)
@click.option(
    "--timesteps", default=16, type=int, help="入力シーケンス長 (SeqLen)。"
)
@click.option(
    "--batch-size", default=2, type=int, help="バッチサイズ (可視化は batch 0 のみ)。"
)
@click.option(
    "--output-prefix",
    default="runs/spike_viz/plot",
    type=str,
    help="出力画像ファイルのプレフィックス (例: runs/viz/micro)。",
)
def spike_visualize(model_config, timesteps, batch_size, output_prefix):
    """
    スパイクのラスタプロット (scripts/visualize_spike_patterns.py) を生成します。
    """
    script_path = "scripts/visualize_spike_patterns.py"
    args = [
        "--model_config", model_config,
        "--timesteps", str(timesteps),
        "--batch_size", str(batch_size),
        "--output_prefix", output_prefix,
    ]
    run_script(script_path, args)

cli.add_command(debug_cli, name="debug")


# --- 13. 知識編集 (Knowledge) ---
# 新機能: 外部記憶(RAG)を直接操作するためのコマンドグループ
@click.group()
def knowledge_cli():
    """外部記憶（知識ベース）の編集と管理"""
    pass

@knowledge_cli.command(name="add")
@click.argument('concept', type=str)
@click.argument('description', type=str)
@click.option('--relation', default='is_defined_as', help="概念と記述の関係性 (例: is_defined_as, causes, related_to)")
@click.option('--vector-store-path', default="runs/vector_store", help="ベクトルストアのパス")
def knowledge_add(concept, description, relation, vector_store_path):
    """
    新しい知識をベクトルストアに追加します。
    
    CONCEPT: 知識の主題 (例: "SNN")\n
    DESCRIPTION: 知識の内容 (例: "省エネで学習可能な次世代AI")
    """
    # 直接Pythonコードを実行してRAGSystemを呼び出す
    code = f"""
from snn_research.cognitive_architecture.rag_snn import RAGSystem
rag = RAGSystem(vector_store_path='{vector_store_path}')
rag.add_relationship('{concept}', '{relation}', '{description}')
"""
    # 一時スクリプトとして実行
    subprocess.run([find_python_executable(), "-c", code], check=True)

@knowledge_cli.command(name="update-causal")
@click.option('--cause', required=True, help="原因となるイベント")
@click.option('--effect', required=True, help="結果となるイベント")
@click.option('--condition', default=None, help="条件 (オプション)")
@click.option('--vector-store-path', default="runs/vector_store", help="ベクトルストアのパス")
def knowledge_update_causal(cause, effect, condition, vector_store_path):
    """
    因果関係をベクトルストアに追加・更新します。
    """
    cond_str = f"'{condition}'" if condition else "None"
    code = f"""
from snn_research.cognitive_architecture.rag_snn import RAGSystem
rag = RAGSystem(vector_store_path='{vector_store_path}')
rag.add_causal_relationship(cause='{cause}', effect='{effect}', condition={cond_str})
"""
    subprocess.run([find_python_executable(), "-c", code], check=True)

@knowledge_cli.command(name="search")
@click.argument('query', type=str)
@click.option('--k', default=3, help="検索する件数")
@click.option('--vector-store-path', default="runs/vector_store", help="ベクトルストアのパス")
def knowledge_search(query, k, vector_store_path):
    """
    知識ベースからクエリに関連する情報を検索します。
    """
    code = f"""
from snn_research.cognitive_architecture.rag_snn import RAGSystem
rag = RAGSystem(vector_store_path='{vector_store_path}')
results = rag.search('{query}', k={k})
print(f"--- Search Results for '{{'{query}'}}' ---")
for i, res in enumerate(results):
    print(f"{{i+1}}. {{res}}")
"""
    subprocess.run([find_python_executable(), "-c", code], check=True)

cli.add_command(knowledge_cli, name="knowledge")


# --- メイン実行 ---
if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        logger.error(f"❌ 致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)