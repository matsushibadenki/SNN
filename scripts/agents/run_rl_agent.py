# ファイルパス: scripts/runners/run_rl_agent.py

from app.containers import TrainingContainer
from pathlib import Path
import argparse  # E402 fixed
import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_rl_agent.py
# Title: 強化学習エージェント実行スクリプト
# Description:
#   BioRLTrainerを使用して強化学習エージェントをトレーニングするスクリプト。

#   ヘルスチェックおよび実験で使用される。

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run Bio-Inspired Reinforcement Learning Agent")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--output_dir", type=str,
                        default="workspace/runs/rl_experiment", help="Directory to save results")
    parser.add_argument(
        "--config", type=str, default="configs/templates/base_config.yaml", help="Base config file")

    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # DIコンテナの初期化
    print("Initializing TrainingContainer...")
    container = TrainingContainer()
    container.config.from_yaml(args.config)

    # トレーナーの取得 (DIコンテナから依存関係解決済みのインスタンスを取得)
    # ここで bio_rl_agent, grid_world_env, bio_learning_rule などが自動的に構築される
    trainer = container.bio_rl_trainer()

    print(f"Starting Bio-RL training for {args.episodes} episodes...")

    # 学習の実行
    results = trainer.train(num_episodes=args.episodes)

    print("Training complete.")
    print(
        f"Final Average Reward: {results.get('final_average_reward', 0.0):.4f}")


if __name__ == "__main__":
    main()
