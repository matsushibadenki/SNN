# snn_research/cli/app_commands.py

import click
import subprocess
from .utils import run_script, run_external_command, verify_path_exists, find_python_executable

# --- UI ---


@click.group(name="ui")
def ui_cli():
    """Gradio UI"""
    pass


@ui_cli.command(name="start")
@click.option('--start-langchain', is_flag=True)
@click.option('--model-config', default="configs/models/small.yaml")
@click.option('--model-path', type=click.Path(), default=None)
@click.option('--cifar_model_config', help="CIFARモデル設定")
def ui_start(start_langchain, model_config, model_path, **kwargs):
    """UI起動"""
    if model_path:
        verify_path_exists(model_path, "モデルファイル",
                           "学習済みモデルのパスを確認するか、先に学習を実行してください。")

    script_path = "app/langchain_main.py" if start_langchain else "app/main.py"
    args = ["--model_config", model_config]
    if model_path:
        args.extend(["--model_path", model_path])
    for key, value in kwargs.items():
        if value:
            args.extend([f"--{key}", value])
    run_script(script_path, args)

# --- エージェント ---


@click.group(name="agent")
def agent_cli():
    """エージェント"""
    pass


@agent_cli.command(name="solve")
@click.option('--task-description', required=True)
@click.option('--unlabeled-data-path', default=None)
@click.option('--force-retrain', is_flag=True)
def agent_solve(task_description, unlabeled_data_path, force_retrain):
    """タスク解決"""
    script_path = "scripts/agents/run_agent.py"
    args = ["--task_description", task_description]
    if unlabeled_data_path:
        args.extend(["--unlabeled_data_path", unlabeled_data_path])
    if force_retrain:
        args.append("--force_retrain")
    run_script(script_path, args)


@agent_cli.command(name="rl")
@click.option('--episodes', default=1000)
def agent_rl(episodes):
    """強化学習"""
    script_path = "scripts/agents/run_rl_agent.py"
    args = ["--episodes", str(episodes)]
    run_script(script_path, args)


@agent_cli.command(name="planner")
@click.option('--task-request', required=True)
@click.option('--context-data', required=True)
def agent_planner(task_request, context_data):
    """プランナー"""
    script_path = "scripts/agents/run_planner.py"
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
        script_path = "scripts/visualization/observe_brain_thought_process.py"
    else:
        script_path = "scripts/experiments/run_brain_simulation.py"
        args.extend(["--prompt", prompt])
    run_script(script_path, args)


@agent_cli.command(name="life-form")
@click.option('--duration', default=60)
@click.option('--model-config', default='configs/models/small.yaml')
def agent_life_form(duration, model_config):
    """デジタル生命体"""
    script_path = "scripts/agents/run_life_form.py"
    args = ["--duration", str(duration), "--model_config", model_config]
    run_script(script_path, args)

# --- デバッグ ---


@click.group(name="debug")
def debug_cli():
    """デバッグ"""
    pass


@debug_cli.command(name="analyze")
@click.option('--tool', default='all')
@click.option('--skip-mypy', is_flag=True)
@click.option('--skip-flake8', is_flag=True)
def debug_analyze(tool, skip_mypy, skip_flake8):
    # パスが変更されたため、mypy/flake8のターゲットも更新
    targets = ["snn_research", "app", "scripts"]  # scripts全体を対象に含める

    if (tool in ['all', 'flake8']) and not skip_flake8:
        # globで直下は取らなくて良い（ほぼ無くなるため）
        run_external_command(["flake8"] + targets)
    if (tool in ['all', 'mypy']) and not skip_mypy:
        mypy_targets = ["snn_research", "app", "scripts"]
        run_external_command(["mypy"] + mypy_targets)


@debug_cli.command(name="spike-test")
@click.option('--model-config', required=True, type=click.Path(exists=True))
@click.option('--timesteps', default=16)
@click.option('--batch-size', default=4)
def debug_spike_test(model_config, timesteps, batch_size):
    script_path = "scripts/debug/debug_spike_activity.py"
    args = ["--model_config", model_config, "--timesteps",
            str(timesteps), "--batch_size", str(batch_size)]
    run_script(script_path, args)


@debug_cli.command("spike-visualize")
@click.option("--model-config", required=True, type=click.Path(exists=True))
@click.option("--timesteps", default=16, type=int)
@click.option("--batch-size", default=2, type=int)
@click.option("--output-prefix", default="workspace/runs/spike_viz/plot")
def spike_visualize(model_config, timesteps, batch_size, output_prefix):
    script_path = "scripts/visualization/visualize_spike_patterns.py"
    args = ["--model_config", model_config, "--timesteps",
            str(timesteps), "--batch_size", str(batch_size), "--output_prefix", output_prefix]
    run_script(script_path, args)

# --- 知識編集 ---


@click.group(name="knowledge")
def knowledge_cli():
    """知識編集"""
    pass


@knowledge_cli.command(name="add")
@click.argument('concept')
@click.argument('description')
@click.option('--relation', default='is_defined_as')
@click.option('--vector-store-path', default="workspace/runs/vector_store")
def knowledge_add(concept, description, relation, vector_store_path):
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); rag.add_relationship('{concept}', '{relation}', '{description}')"
    subprocess.run([find_python_executable(), "-c", code], check=True)


@knowledge_cli.command(name="update-causal")
@click.option('--cause', required=True)
@click.option('--effect', required=True)
@click.option('--condition', default=None)
@click.option('--vector-store-path', default="workspace/runs/vector_store")
def knowledge_update_causal(cause, effect, condition, vector_store_path):
    cond_str = f"'{condition}'" if condition else "None"
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); rag.add_causal_relationship(cause='{cause}', effect='{effect}', condition={cond_str})"
    subprocess.run([find_python_executable(), "-c", code], check=True)


@knowledge_cli.command(name="search")
@click.argument('query')
@click.option('--k', default=3)
@click.option('--vector-store-path', default="workspace/runs/vector_store")
def knowledge_search(query, k, vector_store_path):
    code = f"from snn_research.cognitive_architecture.rag_snn import RAGSystem; rag = RAGSystem(vector_store_path='{vector_store_path}'); results = rag.search('{query}', k={k}); print(results)"
    subprocess.run([find_python_executable(), "-c", code], check=True)


def register_app_commands(cli):
    cli.add_command(ui_cli)
    cli.add_command(agent_cli)
    cli.add_command(debug_cli)
    cli.add_command(knowledge_cli)
