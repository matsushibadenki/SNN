#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ファイルパス: snn-cli.py
タイトル: SNN Project CLI Tool (パス修正版)
機能説明:
SNNプロジェクトの主要なワークフロー（学習、評価、変換、UI起動、最適化、診断など）を
統一的に管理するためのコマンドラインインターフェース（CLI）ツール。
clickライブラリを使用して、モジュール化されたコマンドグループを提供します。
修正: scripts/runners/ ディレクトリ内のスクリプトへのパスを修正。
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
    """
    python_exec = find_python_executable()
    command = [python_exec, script_path] + args
    
    logger.info(f"実行中: {' '.join(command)}")
    
    try:
        if capture_output:
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
    """指定された外部コマンドをサブプロセスとして実行する。"""
    logger.info(f"実行中: {' '.join(command_list)}")
    try:
        if capture_output:
            result = subprocess.run(command_list, check=False, text=True, capture_output=True)
        else:
            result = subprocess.run(command_list, text=True)
            
        if result.returncode == 0:
            logger.info(f"コマンド {command_list[0]} が正常に完了しました。")
        else:
            logger.warning(f"コマンド {command_list[0]} が完了しました (リターンコード: {result.returncode})。")
        return result
    except FileNotFoundError:
        logger.error(f"コマンドが見つかりません: {command_list[0]}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"コマンド {command_list[0]} 実行中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. メインCLIグループ ---
@click.group()
def cli():
    """SNNプロジェクト 統合CLIツール (v11.1)"""
    pass

# --- 2. クリーンアップ機能 ---
@click.group()
def clean_cli():
    """プロジェクトのクリーンアップ"""
    pass

def _perform_clean(delete_models=False, delete_data=False, yes=False):
    targets = ['runs', 'workspace', 'results']
    if delete_data: targets.append('precomputed_data')
    protected_extensions = {'.md'}
    if not delete_data: protected_extensions.update({'.jsonl', '.json', '.db', '.csv'})
    if not delete_models: protected_extensions.update({'.pth', '.pt', '.safetensors'})

    files_to_delete = []
    for target_dir in targets:
        if os.path.isdir(target_dir):
            for filepath in glob(os.path.join(target_dir, '**', '*'), recursive=True):
                if os.path.isfile(filepath):
                    file_ext = os.path.splitext(filepath)[1]
                    if file_ext not in protected_extensions:
                        files_to_delete.append(filepath)

    if not files_to_delete:
        logger.info("削除対象のファイルが見つかりません。")
    else:
        logger.info("以下のファイルが削除されます:")
        for f in files_to_delete[:20]: logger.info(f"  {f}")
        if len(files_to_delete) > 20: logger.info(f"  ...他 {len(files_to_delete) - 20} ファイル")

        if not yes: click.confirm(f"{len(files_to_delete)}個のファイルを削除してもよろしいですか？", abort=True)
        
        deleted_count = 0
        for f in files_to_delete:
            try:
                os.remove(f)
                deleted_count += 1
            except OSError as e: logger.warning(f"ファイルを削除できませんでした: {f} (エラー: {e})")
        logger.info(f"{deleted_count} / {len(files_to_delete)} 個のファイルを削除しました。")

    # 空ディレクトリ削除
    removed_dirs_count = 0
    for target_dir in targets:
        if os.path.isdir(target_dir):
            for root, dirs, files in os.walk(target_dir, topdown=False):
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            removed_dirs_count += 1
                    except OSError: pass
    if removed_dirs_count > 0: logger.info(f"{removed_dirs_count} 個の空ディレクトリを削除しました。")

@clean_cli.command(name="logs")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
def clean_logs(yes):
    """ログとキャッシュのみ削除"""
    _perform_clean(delete_models=False, delete_data=False, yes=yes)

@clean_cli.command(name="models")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
def clean_models(yes):
    """ログ、キャッシュ、モデルを削除"""
    _perform_clean(delete_models=True, delete_data=False, yes=yes)

@clean_cli.command(name="all")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
def clean_all(yes):
    """すべて削除"""
    _perform_clean(delete_models=True, delete_data=True, yes=yes)

cli.add_command(clean_cli, name="clean")

# --- 3. HPO / チューニング ---
@click.group()
def hpo_cli():
    """ハイパーパラメータ最適化"""
    pass

@hpo_cli.command(name="run")
@click.argument('model_config', type=click.Path(exists=True))
@click.argument('task_name', type=str)
@click.option('--target-script', default="scripts/runners/train.py", help="最適化対象スクリプト。") # 修正
@click.option('--teacher-model', default="models/ann_teacher.pth", help="教師モデルパス。")
@click.option('--n-trials', default=50, help="試行回数。")
@click.option('--eval-epochs', default=5, help="評価エポック数。")
@click.option('--metric-name', default="accuracy", help="最適化メトリクス。")
@click.option('--storage', default=None, help="OptunaストレージURL。")
@click.option('--study-name', default=None, help="Study名。")
def hpo_run(model_config, task_name, target_script, teacher_model, n_trials, eval_epochs, metric_name, storage, study_name):
    """run_hpo.py を実行"""
    # パス修正: scripts/runners/run_hpo.py
    script_path = "scripts/runners/run_hpo.py"
    args = [
        "--model_config", model_config,
        "--task", task_name,
        "--target_script", target_script,
        "--teacher_model", teacher_model,
        "--n-trials", str(n_trials),
        "--eval-epochs", str(eval_epochs),
        "--metric-name", metric_name,
    ]
    if storage: args.extend(["--storage", storage])
    if study_name: args.extend(["--study_name", study_name])
    run_script(script_path, args)

@hpo_cli.command(name="efficiency")
@click.option('--n-trials', default=20)
@click.option('--config', default="configs/experiments/smoke_test_config.yaml")
@click.option('--model-config', default="configs/models/micro.yaml")
@click.option('--data-path', default="data/smoke_test_data.jsonl")
def hpo_efficiency(n_trials, config, model_config, data_path):
    """自動チューニング実行"""
    script_path = "scripts/auto_tune_efficiency.py"
    args = ["--n-trials", str(n_trials), "--config", config, "--model-config", model_config, "--data-path", data_path]
    run_script(script_path, args)

cli.add_command(hpo_cli, name="hpo")
cli.add_command(hpo_cli, name="tune")

# --- 4. UI ---
@click.group()
def ui_cli():
    """Gradio UI"""
    pass

@ui_cli.command(name="start")
@click.option('--start-langchain', is_flag=True)
@click.option('--model-config', default="configs/models/small.yaml")
@click.option('--model-path', type=click.Path(exists=True), default=None)
@click.option('--cifar_model_config', help="CIFARモデル設定")
def ui_start(start_langchain, model_config, model_path, **kwargs):
    """UI起動"""
    script_path = "app/langchain_main.py" if start_langchain else "app/main.py"
    args = ["--model_config", model_config]
    if model_path: args.extend(["--model_path", model_path])
    for key, value in kwargs.items():
        if value: args.extend([f"--{key}", value])
    run_script(script_path, args)

cli.add_command(ui_cli, name="ui")

# --- 5. モデル学習 ---
@click.group()
def train_cli():
    """SNNモデル学習"""
    pass

@train_cli.command(name="gradient")
@click.option('--config', default="configs/experiments/cifar10_spikingcnn_config.yaml")
@click.option('--model-config', required=True)
@click.option('--data-path', default="data/")
@click.option('--override-config', default=None)
@click.option('--resume-path', default=None, type=click.Path(exists=True))
@click.option('--distributed', is_flag=True)
@click.option('--task-name', default=None)
@click.option('--load-ewc-data', default=None)
def gradient_train(config, model_config, data_path, override_config, resume_path, distributed, task_name, load_ewc_data):
    """代理勾配法による学習"""
    # パス修正: scripts/runners/train.py
    script_path = "scripts/runners/train.py"
    args = ["--config", config, "--model_config", model_config, "--data_path", data_path]
    if override_config: args.extend(["--override_config", override_config])
    if resume_path: args.extend(["--resume_path", resume_path])
    if distributed: args.append("--distributed")
    if task_name: args.extend(["--task_name", task_name])
    if load_ewc_data: args.extend(["--load_ewc_data", load_ewc_data])
    run_script(script_path, args)

@train_cli.command(name="distill")
@click.option('--task', required=True)
@click.option('--teacher-model', required=True)
@click.option('--model-config', required=True)
@click.option('--epochs', default=15)
def train_distill(task, teacher_model, model_config, epochs):
    """
    知識蒸留 (train.py を蒸留モードで実行)
    """
    # パス修正: scripts/runners/train.py を使用
    script_path = "scripts/runners/train.py"
    # 蒸留用の設定をオーバーライドで渡す
    args = [
        "--config", "configs/templates/base_config.yaml", # ベース設定
        "--model_config", model_config,
        "--paradigm", "gradient_based",
        "--override_config", "training.gradient_based.type=distillation",
        "--override_config", f"training.gradient_based.distillation.teacher_model={teacher_model}",
        "--override_config", f"training.epochs={epochs}",
        # データパスはタスクに応じて適切に設定する必要があるが、ここでは簡易的にデフォルトを使用するか
        # ユーザーに追加引数を求めるのが望ましい。今回は必須引数 task を task_name として渡す
        "--task_name", task
    ]
    run_script(script_path, args)

cli.add_command(train_cli, name="train")

# --- 6. ベンチマーク ---
@click.group()
def benchmark_cli():
    """ベンチマーク"""
    pass

@benchmark_cli.command(name="run")
@click.option('--experiment', required=True)
@click.option('--epochs', default=5)
@click.option('--tag', default="BenchmarkRun")
@click.option('--model-config', default=None) # 追加
def benchmark_run(experiment, epochs, tag, model_config):
    """ベンチマーク実行"""
    script_path = "scripts/run_benchmark_suite.py"
    args = ["--experiment", experiment, "--epochs", str(epochs), "--tag", tag]
    if model_config:
        args.extend(["--model_config", model_config])
    run_script(script_path, args)

@benchmark_cli.command(name="evaluate-accuracy")
@click.option('--model-path', required=True, type=click.Path(exists=True))
@click.option('--model-config', required=True)
@click.option('--model-type', required=True, type=click.Choice(['snn', 'ann']))
@click.option('--experiment', required=True)
@click.option('--tag', default="AccuracyEvaluation")
def evaluate_accuracy(model_path, model_config, model_type, experiment, tag):
    """精度評価"""
    script_path = "scripts/run_benchmark_suite.py"
    args = ["--eval_only", "--model_path", model_path, "--model_config", model_config, "--model_type", model_type, "--experiment", experiment, "--tag", tag]
    run_script(script_path, args)

@benchmark_cli.command(name="continual")
@click.option('--epochs-task-a', default=3)
@click.option('--epochs-task-b', default=3)
def benchmark_continual(epochs_task_a, epochs_task_b):
    """継続学習実験"""
    script_path = "scripts/run_continual_learning_experiment.py"
    args = ["--epochs_task_a", str(epochs_task_a), "--epochs_task_b", str(epochs_task_b)]
    run_script(script_path, args)

cli.add_command(benchmark_cli, name="benchmark")

# --- 7. モデル変換 ---
@click.group()
def convert_cli():
    """モデル変換"""
    pass

@convert_cli.command(name="ann2snn-cnn")
@click.argument('ann_model_path', type=click.Path(exists=True))
@click.argument('output_snn_path', type=click.Path())
@click.option('--snn-model-config', required=True)
def convert_ann2snn_cnn(ann_model_path, output_snn_path, snn_model_config):
    """CNN変換"""
    script_path = "scripts/convert_model.py"
    args = ["--ann_model_path", ann_model_path, "--output_snn_path", output_snn_path, "--method", "cnn-convert", "--snn_model_config", snn_model_config]
    run_script(script_path, args)

cli.add_command(convert_cli, name="convert")

# --- 8. エージェント ---
@click.group()
def agent_cli():
    """エージェント"""
    pass

@agent_cli.command(name="solve")
@click.option('--task-description', required=True)
@click.option('--unlabeled-data-path', default=None)
@click.option('--force-retrain', is_flag=True)
def agent_solve(task_description, unlabeled_data_path, force_retrain):
    """タスク解決"""
    # パス修正
    script_path = "scripts/runners/run_agent.py"
    args = ["--task_description", task_description]
    if unlabeled_data_path: args.extend(["--unlabeled_data_path", unlabeled_data_path])
    if force_retrain: args.append("--force_retrain")
    run_script(script_path, args)

# agent evolve コマンドはスクリプトが存在しないため削除

@agent_cli.command(name="rl")
@click.option('--episodes', default=1000)
def agent_rl(episodes):
    """強化学習"""
    # パス修正
    script_path = "scripts/runners/run_rl_agent.py"
    args = ["--episodes", str(episodes)]
    run_script(script_path, args)

@agent_cli.command(name="planner")
@click.option('--task-request', required=True)
@click.option('--context-data', required=True)
def agent_planner(task_request, context_data):
    """プランナー"""
    # パス修正
    script_path = "scripts/runners/run_planner.py"
    args = ["--task_request", task_request, "--context_data", context_data]
    run_script(script_path, args)

@agent_cli.command(name="brain")
@click.option('--prompt', default="今日の天気は？")
@click.option('--loop', is_flag=True)
@click.option('--model-config', default='configs/models/small.yaml')
def agent_brain(prompt, loop, model_config):
    """人工脳シミュレーション"""
    args = ["--model_config", model_config]
    if loop:
        script_path = "scripts/observe_brain_thought_process.py"
    else:
        # パス修正
        script_path = "scripts/runners/run_brain_simulation.py"
        args.extend(["--prompt", prompt])
    run_script(script_path, args)

@agent_cli.command(name="life-form")
@click.option('--duration', default=60)
@click.option('--model-config', default='configs/models/small.yaml')
def agent_life_form(duration, model_config):
    """デジタル生命体"""
    # パス修正
    script_path = "scripts/runners/run_life_form.py"
    args = ["--duration", str(duration), "--model_config", model_config]
    run_script(script_path, args)

cli.add_command(agent_cli, name="agent")

# --- 9. 診断 ---
@click.group()
def diagnostics_cli():
    """診断"""
    pass

@diagnostics_cli.command(name="report-efficiency")
@click.option('--model-config', required=True)
@click.option('--data-path', default="data/cifar10")
@click.option('--model-path', type=click.Path(exists=True))
def report_efficiency(model_config, data_path, model_path):
    """効率診断"""
    script_path = "scripts/report_sparsity_and_T.py"
    args = ["--model_config", model_config, "--data_path", data_path]
    if model_path: args.extend(["--model_path", model_path])
    run_script(script_path, args)

cli.add_command(diagnostics_cli, name="diagnostics")

# --- 10. ヘルスチェック ---
@cli.command(name="health-check")
def health_check():
    """健全性チェック"""
    script_path = "scripts/run_project_health_check.py"
    run_script(script_path, [])

# --- 11. データ管理 ---
@click.group()
def data_cli():
    """データ管理"""
    pass

@data_cli.command(name="prepare")
@click.option('--dataset', required=True)
@click.option('--output-dir', default='data/prepared')
def data_prepare(dataset, output_dir):
    script_path = "scripts/data_preparation.py"
    args = ["--dataset", dataset, "--output_dir", output_dir]
    run_script(script_path, args)

@data_cli.command(name="build-kb")
@click.option('--input-dir', default='data/knowledge_source')
@click.option('--output-db', default='workspace/knowledge_base.db')
def data_build_kb(input_dir, output_db):
    script_path = "scripts/build_knowledge_base.py"
    args = ["--input_dir", input_dir, "--output_db", output_db]
    run_script(script_path, args)

@data_cli.command(name="prep-distill")
@click.option('--task', required=True)
@click.option('--teacher-model', required=True)
@click.option('--output-dir', default='precomputed_data/distillation')
def data_prep_distill(task, teacher_model, output_dir):
    script_path = "scripts/prepare_distillation_data.py"
    args = ["--task", task, "--teacher_model", teacher_model, "--output_dir", output_dir]
    run_script(script_path, args)

cli.add_command(data_cli, name="data")

# --- 12. デバッグ ---
@click.group()
def debug_cli():
    """デバッグ"""
    pass

@debug_cli.command(name="analyze")
@click.option('--tool', default='all')
@click.option('--skip-mypy', is_flag=True)
@click.option('--skip-flake8', is_flag=True)
def debug_analyze(tool, skip_mypy, skip_flake8):
    targets = ["snn_research", "app", "scripts", "tests"] + glob("*.py")
    if (tool in ['all', 'flake8']) and not skip_flake8:
        run_external_command(["flake8"] + targets)
    if (tool in ['all', 'mypy']) and not skip_mypy:
        # パス修正: scripts/runners を追加
        mypy_targets = ["snn_research", "app", "scripts", "tests", "snn-cli.py", "scripts/runners/train.py"]
        run_external_command(["mypy"] + mypy_targets)

@debug_cli.command(name="spike-test")
@click.option('--model-config', required=True, type=click.Path(exists=True))
@click.option('--timesteps', default=16)
@click.option('--batch-size', default=4)
def debug_spike_test(model_config, timesteps, batch_size):
    script_path = "scripts/debug_spike_activity.py"
    args = ["--model_config", model_config, "--timesteps", str(timesteps), "--batch_size", str(batch_size)]
    run_script(script_path, args)

@debug_cli.command("spike-visualize")
@click.option("--model-config", required=True, type=click.Path(exists=True))
@click.option("--timesteps", default=16, type=int)
@click.option("--batch-size", default=2, type=int)
@click.option("--output-prefix", default="runs/spike_viz/plot")
def spike_visualize(model_config, timesteps, batch_size, output_prefix):
    script_path = "scripts/visualize_spike_patterns.py"
    args = ["--model_config", model_config, "--timesteps", str(timesteps), "--batch_size", str(batch_size), "--output_prefix", output_prefix]
    run_script(script_path, args)

cli.add_command(debug_cli, name="debug")

# --- 13. 知識編集 ---
@click.group()
def knowledge_cli():
    """知識編集"""
    pass

@knowledge_cli.command(name="add")
@click.argument('concept')
@click.argument('description')
@click.option('--relation', default='is_defined_as')
@click.option('--vector-store-path', default="runs/vector_store")
def knowledge_add(concept, description, relation, vector_store_path):
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); rag.add_relationship('{concept}', '{relation}', '{description}')"
    subprocess.run([find_python_executable(), "-c", code], check=True)

@knowledge_cli.command(name="update-causal")
@click.option('--cause', required=True)
@click.option('--effect', required=True)
@click.option('--condition', default=None)
@click.option('--vector-store-path', default="runs/vector_store")
def knowledge_update_causal(cause, effect, condition, vector_store_path):
    cond_str = f"'{condition}'" if condition else "None"
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); rag.add_causal_relationship(cause='{cause}', effect='{effect}', condition={cond_str})"
    subprocess.run([find_python_executable(), "-c", code], check=True)

@knowledge_cli.command(name="search")
@click.argument('query')
@click.option('--k', default=3)
@click.option('--vector-store-path', default="runs/vector_store")
def knowledge_search(query, k, vector_store_path):
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); results = rag.search('{query}', k={k}); print(results)"
    subprocess.run([find_python_executable(), "-c", code], check=True)

cli.add_command(knowledge_cli, name="knowledge")

if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        logger.error(f"❌ 致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
