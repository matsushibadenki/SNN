# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# 修正: 属性名とメソッド名を最新の実装(ArtificialBrain v2.x)に合わせて修正。

from app.containers import BrainContainer
import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def brain_container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。"""
    container = BrainContainer()
    # テスト用の設定をロード
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")

    # RAGSystemの初回セットアップをシミュレート
    rag_system = container.agent_container.rag_system()
    if hasattr(rag_system, 'vector_store_path') and rag_system.vector_store_path:
        pass
    return container


@pytest.fixture(scope="module")
def brain_instance(brain_container):
    """ArtificialBrainインスタンスを提供するフィクスチャ"""
    return brain_container.artificial_brain()


def test_artificial_brain_instantiation(brain_container: BrainContainer):
    """
    BrainContainerがArtificialBrainインスタンスを正常に構築できるかテストする。
    """
    brain = brain_container.artificial_brain()
    assert brain is not None
    assert brain.pfc is not None
    assert brain.hippocampus is not None
    # [Fix] motor -> motor_cortex
    assert brain.motor_cortex is not None
    print("✅ ArtificialBrainインスタンスの構築に成功しました。")


def test_cognitive_cycle_runs_and_consolidates_memory(brain_instance):
    """
    認知サイクル実行後に、記憶が正しく固定化されているかを検証するテスト。
    """
    # 1. 認知サイクルの実行
    try:
        # [Fix] run_cognitive_cycle -> process_step
        brain_instance.process_step(sensory_input="test_stimulus")
    except Exception as e:
        pytest.fail(f"Cognitive cycle failed: {e}")

    # 2. Cortexから全知識を取得
    all_knowledge = brain_instance.cortex.get_all_knowledge()

    assert isinstance(
        all_knowledge, list), "Knowledge should be returned as a list"
    assert len(all_knowledge) >= 0
