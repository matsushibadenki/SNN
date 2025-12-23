# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# (修正: mypy [index] エラー修正)
# Title: 人工脳 統合テスト
# ...

import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.containers import BrainContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

@pytest.fixture(scope="module")
def brain_container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。"""
    container = BrainContainer()
    # テスト用の設定をロード
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")
    
    # RAGSystemの初回セットアップをシミュレート
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        rag_system.setup_vector_store()
    return container

def test_artificial_brain_instantiation(brain_container: BrainContainer):
    """
    BrainContainerがArtificialBrainインスタンスを正常に構築できるかテストする。
    """
    brain = brain_container.artificial_brain()
    assert brain is not None
    assert brain.pfc is not None
    assert brain.hippocampus is not None
    assert brain.motor is not None
    print("✅ ArtificialBrainインスタンスの構築に成功しました。")

def test_cognitive_cycle_runs_and_consolidates_memory(brain_instance):
    """
    認知サイクル実行後に、記憶が正しく固定化されているかを検証するテスト。
    """
    # 1. 認知サイクルの実行
    brain_instance.run_cognitive_cycle(sensory_input="test_stimulus")
    
    # 2. Cortexから全知識を取得
    # Cortex.get_all_knowledge() は list[str] を返す
    all_knowledge = brain_instance.cortex.get_all_knowledge()
    
    # [Fix] list型には .get() がないため、直接リストの中身を確認するように修正
    assert isinstance(all_knowledge, list), "Knowledge should be returned as a list"
    
    # キーワードが含まれているか確認
    found_test_knowledge = any("test_stimulus" in k for k in all_knowledge)
    assert found_test_knowledge, "Consolidated knowledge should contain the test stimulus"
    
    # 知識の数を確認
    assert len(all_knowledge) >= 0
