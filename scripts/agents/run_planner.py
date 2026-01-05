# ファイルパス: scripts/runners/run_planner.py

from app.containers import AgentContainer
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

# matsushibadenki/snn4/snn4-main/run_planner.py
# Phase 3: 高次認知アーキテクチャの実行インターフェース
# 改善点: プランナーに必要な依存関係(RAGSystem, ModelRegistry)を初期化するように修正。

# 修正点: 依存関係の二重初期化をなくし、DIコンテナで生成されたインスタンスを正しく使用するように修正。


def main():
    """
    階層的思考プランナーに複雑なタスクを依頼し、
    複数の専門家SNNを連携させた問題解決を実行させる。
    """
    parser = argparse.ArgumentParser(
        description="SNN高次認知アーキテクチャ 実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_request",
        type=str,
        required=True,
        help="エージェントに解決させたい、自然言語による複雑なタスク要求。\n例: 'この記事を要約して、その内容の感情を分析してください。'"
    )
    parser.add_argument(
        "--context_data",
        type=str,
        required=True,
        help="タスク処理の対象となる文脈データ（文章や質問など）。"
    )

    args = parser.parse_args()

    # DIコンテナを初期化し、依存関係が注入済みのコンポーネントを取得
    container = AgentContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")

    planner = container.hierarchical_planner()
    rag_system = container.rag_system()

    # RAGの知識ベースを構築（存在しない場合）
    if not rag_system.vector_store:
        print("知識ベースが存在しないため、初回構築を行います...")
        rag_system.setup_vector_store()

    # --- ◾️◾️◾️◾️◾️↓修正↓◾️◾️◾️◾️◾️ ---
    # 依存関係の重複した手動初期化を削除。コンテナから取得したものを利用する。
    # --- ◾️◾️◾️◾️◾️↑修正↑◾️◾️◾️◾️◾️ ---

    # --- プランナーにタスク処理を依頼 ---
    final_result = planner.execute_task(
        task_request=args.task_request,
        context=args.context_data
    )

    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
        print("="*56)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。必要な専門家モデルが不足している可能性があります。")


if __name__ == "__main__":
    main()
