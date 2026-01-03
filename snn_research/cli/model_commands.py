#snn_research/cli/model_commands.py

import click
import os
from .utils import run_script, verify_path_exists, logger

# --- HPO / チューニング ---
@click.group(name="hpo")
def hpo_cli():
    """ハイパーパラメータ最適化"""
    pass

@hpo_cli.command(name="run")
@click.argument('model_config', type=click.Path(exists=True))
@click.argument('task_name', type=str)
@click.option('--target-script', default="scripts/training/train.py", help="最適化対象スクリプト。")
@click.option('--teacher-model', default="gpt2", help="教師モデルパス。")
@click.option('--n-trials', default=50, help="試行回数。")
@click.option('--eval-epochs', default=5, help="評価エポック数。")
@click.option('--metric-name', default="accuracy", help="最適化メトリクス。")
@click.option('--storage', default=None, help="OptunaストレージURL。")
@click.option('--study-name', default=None, help="Study名。")
def hpo_run(model_config, task_name, target_script, teacher_model, n_trials, eval_epochs, metric_name, storage, study_name):
    """run_hpo.py を実行"""
    script_path = "scripts/optimization/run_hpo.py"
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
    script_path = "scripts/optimization/auto_tune_efficiency.py"
    args = ["--n-trials", str(n_trials), "--config", config, "--model-config", model_config, "--data-path", data_path]
    run_script(script_path, args)

# --- モデル学習 ---
@click.group(name="train")
def train_cli():
    """SNNモデル学習"""
    pass

@train_cli.command(name="gradient")
@click.option('--config', default="configs/templates/base_config.yaml")
@click.option('--model-config', required=True)
@click.option('--data-path', default="data/smoke_test_data.jsonl")
@click.option('--override-config', default=None)
@click.option('--resume-path', default=None, type=click.Path())
@click.option('--distributed', is_flag=True)
@click.option('--task-name', default=None)
@click.option('--load-ewc-data', default=None)
@click.option('--epochs', default=None)
def gradient_train(config, model_config, data_path, override_config, resume_path, distributed, task_name, load_ewc_data, epochs):
    """代理勾配法による学習"""
    if resume_path:
        verify_path_exists(resume_path, "再開用チェックポイント", "正しいパスを指定してください。")

    script_path = "scripts/training/train.py"
    args = ["--config", config, "--model_config", model_config, "--data_path", data_path]
    if override_config: args.extend(["--override_config", override_config])
    if epochs: args.extend(["--override_config", f"training.epochs={epochs}"])
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
    """知識蒸留"""
    script_path = "scripts/training/train.py"
    args = [
        "--config", "configs/templates/base_config.yaml",
        "--model_config", model_config,
        "--paradigm", "gradient_based",
        "--override_config", "training.gradient_based.type=distillation",
        "--override_config", f"training.gradient_based.distillation.teacher_model={teacher_model}",
        "--override_config", f"training.epochs={epochs}",
        "--task_name", task
    ]
    run_script(script_path, args)

# --- ベンチマーク ---
@click.group(name="benchmark")
def benchmark_cli():
    """ベンチマーク"""
    pass

@benchmark_cli.command(name="run")
@click.option('--experiment', required=True)
@click.option('--epochs', default=5)
@click.option('--tag', default="BenchmarkRun")
@click.option('--model-config', default=None)
def benchmark_run(experiment, epochs, tag, model_config):
    """ベンチマーク実行"""
    script_path = "scripts/benchmarks/run_benchmark_suite.py"
    args = ["--experiment", experiment, "--epochs", str(epochs), "--tag", tag]
    if model_config:
        args.extend(["--model_config", model_config])
    run_script(script_path, args)

@benchmark_cli.command(name="evaluate-accuracy")
@click.option('--model-path', required=True, type=click.Path())
@click.option('--model-config', required=True)
@click.option('--model-type', required=True, type=click.Choice(['SNN', 'ANN']))
@click.option('--experiment', required=True)
@click.option('--tag', default="AccuracyEvaluation")
def evaluate_accuracy(model_path, model_config, model_type, experiment, tag):
    """精度評価"""
    suggestion = None
    if "converted" in model_path:
        suggestion = f"モデルが見つかりません。変換コマンドを実行しましたか？\n例: snn-cli convert ann2snn-cnn {experiment}_ann.pth {model_path} --snn-model-config {model_config}"
    else:
        suggestion = "学習済みモデルのパスが正しいか確認してください。"
    
    verify_path_exists(model_path, "評価対象モデル", suggestion)

    script_path = "scripts/benchmarks/run_benchmark_suite.py"
    args = ["--eval_only", "--model_path", model_path, "--model_config", model_config, "--model_type", model_type, "--experiment", experiment, "--tag", tag]
    run_script(script_path, args)

@benchmark_cli.command(name="continual")
@click.option('--epochs-task-a', default=3)
@click.option('--epochs-task-b', default=3)
def benchmark_continual(epochs_task_a, epochs_task_b):
    """継続学習実験"""
    script_path = "scripts/experiments/run_continual_learning_experiment.py"
    args = ["--epochs_task_a", str(epochs_task_a), "--epochs_task_b", str(epochs_task_b)]
    run_script(script_path, args)

# --- モデル変換 ---
@click.group(name="convert")
def convert_cli():
    """モデル変換"""
    pass

@convert_cli.command(name="ann2snn-cnn")
@click.argument('ann_model_path', type=click.Path(exists=True))
@click.argument('output_snn_path', type=click.Path())
@click.option('--snn-model-config', required=True)
def convert_ann2snn_cnn(ann_model_path, output_snn_path, snn_model_config):
    """CNN変換"""
    output_dir = os.path.dirname(output_snn_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"📂 出力ディレクトリを作成します: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    script_path = "scripts/utils/convert_model.py"
    args = ["--ann_model_path", ann_model_path, "--output_snn_path", output_snn_path, "--method", "cnn-convert", "--snn_model_config", snn_model_config]
    run_script(script_path, args)

# --- 診断 ---
@click.group(name="diagnostics")
def diagnostics_cli():
    """診断"""
    pass

@diagnostics_cli.command(name="report-efficiency")
@click.option('--model-config', required=True)
@click.option('--data-path', default="data/cifar10")
@click.option('--model-path', type=click.Path())
def report_efficiency(model_config, data_path, model_path):
    """効率診断"""
    if model_path:
        verify_path_exists(
            model_path, 
            "診断対象モデル", 
            "モデルパスを指定しない場合、ランダム初期化モデルで効率を計測するか、あるいは正しいパスを指定してください。"
        )

    script_path = "scripts/utils/report_sparsity_and_T.py"
    args = ["--model_config", model_config, "--data_path", data_path]
    if model_path: args.extend(["--model_path", model_path])
    run_script(script_path, args)

def register_model_commands(cli):
    cli.add_command(hpo_cli)
    # alias
    cli.add_command(hpo_cli, name="tune")
    cli.add_command(train_cli)
    cli.add_command(benchmark_cli)
    cli.add_command(convert_cli)
    cli.add_command(diagnostics_cli)