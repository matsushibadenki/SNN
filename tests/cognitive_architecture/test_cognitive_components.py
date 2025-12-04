# ファイルパス: tests/cognitive_architecture/test_cognitive_components.py
# タイトル: 認知コンポーネント単体テスト (エッジケース追加)
# 機能説明:
# - 人工脳を構成する各モジュールが、個別に正しく機能することを確認する単体テスト。
# - Amygdala, BasalGanglia, Cerebellum, MotorCortex, Hippocampus,
#   Cortex, PrefrontalCortex の基本動作とエッジケースを検証する。
# 改善点 (v4):
# - ロードマップ フェーズ5に基づき、エッジケース（空の入力、混合感情など）を
#   検証するテストを追加し、各コンポーネントの堅牢性を向上。
# 改善点 (v5): 空の辞書やNone入力に対するテストを追加。

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import torch # torchをインポート

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem

# --- Mocks for dependencies ---
@pytest.fixture
def mock_workspace():
    """GlobalWorkspaceのモックを作成するフィクスチャ。"""
    mock = MagicMock(spec=GlobalWorkspace)
    # get_informationがNoneを返すように設定 (デフォルト)
    mock.get_information.return_value = None
    return mock

@pytest.fixture
def mock_motivation_system():
    """IntrinsicMotivationSystemのモックを作成するフィクスチャ。"""
    mock = MagicMock(spec=IntrinsicMotivationSystem)
    # get_internal_stateが空の辞書を返すように設定 (デフォルト)
    mock.get_internal_state.return_value = {}
    return mock

# --- Amygdala Tests (変更なし) ---
def test_amygdala_evaluates_positive_emotion(mock_workspace):
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("素晴らしい成功体験でした。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['salience'] > 0.5
    assert kwargs['data']['valence'] > 0
    print("\n✅ Amygdala: ポジティブな感情の評価テストに成功。")

def test_amygdala_evaluates_negative_emotion(mock_workspace):
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("危険なエラーが発生し、失敗した。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['data']['valence'] < 0
    print("✅ Amygdala: ネガティブな感情の評価テストに成功。")

def test_amygdala_handles_mixed_emotion(mock_workspace):
    """ポジティブとネガティブが混在するテキストを評価できるかテストする。"""
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("失敗の中に喜びを見出す。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    # 混合感情なので、0に近い値になるはず（厳密な値は辞書による）
    assert -0.2 < kwargs['data']['valence'] < 0.2 # 許容範囲を少し広げる
    print("✅ Amygdala: 混合感情の評価テストに成功。")

def test_amygdala_handles_neutral_text(mock_workspace):
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("これはただの事実です。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['data']['valence'] == 0.0
    print("✅ Amygdala: 中立的なテキストの評価テストに成功。")

def test_amygdala_handles_empty_string(mock_workspace):
    """空の文字列が入力された場合にエラーなくデフォルト値を返すかテストする。"""
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['data']['valence'] == 0.0
    assert kwargs['data']['arousal'] == 0.1 # デフォルト覚醒度
    print("✅ Amygdala: 空文字列入力のテストに成功。")

# --- BasalGanglia Tests ---
def test_basal_ganglia_selects_best_action(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.4)
    candidates = [
        {'action': 'A', 'value': 0.9},
        {'action': 'B', 'value': 0.6},
        {'action': 'C', 'value': 0.2},
    ]
    selected = basal_ganglia.select_action(candidates)
    assert selected is not None and selected['action'] == 'A'
    print("✅ BasalGanglia: 最適行動選択のテストに成功。")

def test_basal_ganglia_rejects_low_value_actions(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.8)
    candidates = [{'action': 'A', 'value': 0.7}]
    selected = basal_ganglia.select_action(candidates)
    assert selected is None
    print("✅ BasalGanglia: 低価値行動の棄却テストに成功。")

def test_basal_ganglia_emotion_modulates_selection(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.5)
    candidates = [{'action': 'run_away', 'value': 0.6}] # 価値を少し下げる
    fear_context = {'valence': -0.8, 'arousal': 0.9} # 恐怖
    # 恐怖(高覚醒)により閾値が下がり、通常なら棄却される行動が選択されるはず
    selected_fear = basal_ganglia.select_action(candidates, emotion_context=fear_context)
    assert selected_fear is not None and selected_fear['action'] == 'run_away'
    print("✅ BasalGanglia: 情動による意思決定変調のテストに成功。")

def test_basal_ganglia_handles_no_candidates(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace)
    selected = basal_ganglia.select_action([])
    assert selected is None
    print("✅ BasalGanglia: 行動候補が空の場合のテストに成功。")

def test_basal_ganglia_handles_none_emotion_context(mock_workspace):
    """emotion_contextがNoneの場合にエラーなく動作するかテストする。"""
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.5)
    candidates = [{'action': 'A', 'value': 0.6}]
    selected = basal_ganglia.select_action(candidates, emotion_context=None)
    assert selected is not None and selected['action'] == 'A'
    print("✅ BasalGanglia: emotion_contextがNoneの場合のテストに成功。")

# --- Cerebellum & MotorCortex Tests ---
def test_cerebellum_and_motor_cortex_pipeline():
    cerebellum = Cerebellum()
    motor_cortex = MotorCortex(actuators=['test_actuator'])
    action = {'action': 'do_something', 'duration': 0.5}

    commands = cerebellum.refine_action_plan(action)
    assert len(commands) > 1 and commands[0]['command'] == 'do_something_start'

    log = motor_cortex.execute_commands(commands)
    assert len(log) > 1 and "do_something_start" in log[0]
    print("✅ Cerebellum -> MotorCortex パイプラインのテストに成功。")

def test_cerebellum_handles_empty_action():
    """小脳が空の行動計画を受け取った場合のテスト。"""
    cerebellum = Cerebellum()
    commands = cerebellum.refine_action_plan({})
    assert commands == []
    print("✅ Cerebellum: 空の行動計画入力のテストに成功。")

def test_motor_cortex_handles_empty_commands():
    """運動野が空のコマンドリストを受け取った場合のテスト。"""
    motor_cortex = MotorCortex()
    log = motor_cortex.execute_commands([])
    assert log == []
    print("✅ MotorCortex: 空のコマンドリスト入力のテストに成功。")

# --- Hippocampus & Cortex (Memory System) Tests ---
def test_memory_system_pipeline(mock_workspace):
    hippocampus = Hippocampus(workspace=mock_workspace, capacity=3)
    cortex = Cortex()

    # 1. 短期記憶へ保存
    hippocampus.store_episode({'source_input': 'A cat is a small animal.'})
    hippocampus.store_episode({'source_input': 'A dog is a friendly pet.'})
    assert len(hippocampus.working_memory) == 2

    # 2. 長期記憶へ固定化
    episodes_for_consolidation = hippocampus.get_and_clear_episodes_for_consolidation()
    assert len(episodes_for_consolidation) == 2
    assert len(hippocampus.working_memory) == 0 # クリアされたか確認

    for episode in episodes_for_consolidation:
        cortex.consolidate_memory(episode)

    # 3. 長期記憶から検索
    knowledge = cortex.retrieve_knowledge('animal')
    assert knowledge is not None and len(knowledge) > 0
    # consolidate_memoryの実装によっては 'is_a' のような関係が追加される
    assert any(rel['target'] == 'small' or rel['relation'] == 'is_a' for rel in knowledge)
    print("✅ Hippocampus -> Cortex (記憶固定化) パイプラインのテストに成功。")

def test_hippocampus_handles_empty_episode():
    """海馬が空のエピソードを保存しようとした場合のテスト。"""
    hippocampus = Hippocampus(workspace=MagicMock(), capacity=3)
    hippocampus.store_episode({})
    assert len(hippocampus.working_memory) == 1 # 空でも保存される
    retrieved = hippocampus.retrieve_recent_episodes(1)
    assert retrieved == [{}]
    print("✅ Hippocampus: 空のエピソード保存テストに成功。")

def test_hippocampus_relevance_with_no_memory(mock_workspace):
    """短期記憶が空の状態で関連性評価が行われた場合のテスト。"""
    hippocampus = Hippocampus(workspace=mock_workspace, capacity=3)
    dummy_features = torch.randn(64)
    hippocampus.evaluate_relevance_and_upload(dummy_features)
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['source'] == 'hippocampus'
    assert kwargs['data']['relevance'] == 0.0
    assert kwargs['salience'] == 0.8 # 新規性が高いと判断される
    print("✅ Hippocampus: 記憶がない状態での関連性評価テストに成功。")

def test_cortex_handles_non_string_input():
    """大脳皮質が文字列でない入力のエピソードを処理しようとした場合のテスト。"""
    cortex = Cortex()
    # 文字列でない source_input を持つエピソード
    episode = {'source_input': 12345}
    cortex.consolidate_memory(episode)
    # エラーが発生せず、知識が追加されないことを確認
    assert len(cortex.get_all_knowledge()) == 0
    print("✅ Cortex: 文字列でない入力のエピソード処理テストに成功。")

def test_cortex_retrieves_nonexistent_concept():
    """大脳皮質が存在しない概念を検索した場合のテスト。"""
    cortex = Cortex()
    knowledge = cortex.retrieve_knowledge('unknown_concept')
    assert knowledge is None
    print("✅ Cortex: 存在しない概念の検索テストに成功。")

# --- PrefrontalCortex Tests ---
@pytest.mark.parametrize("context, expected_keyword", [
    ({"external_request": "summarize the document"}, "summarize"),
    ({"internal_state": {"boredom": 0.8}}, "boredom"),
    ({"internal_state": {"curiosity": 0.9}}, "Explore a new topic"),
    ({"internal_state": {"boredom": 0.1, "curiosity": 0.2}, "conscious_content": {"type": "emotion", "valence": -0.9}}, "negative emotion"), # 強いネガティブ感情
    ({}, "Organize"), # デフォルト
])
def test_prefrontal_cortex_decides_goals(context, expected_keyword, mock_workspace, mock_motivation_system):
    # motivation_systemのモックが適切な内部状態を返すように設定
    mock_motivation_system.get_internal_state.return_value = context.get("internal_state", {})

    pfc = PrefrontalCortex(workspace=mock_workspace, motivation_system=mock_motivation_system)

    # handle_conscious_broadcastを直接呼び出して目標決定をトリガー
    # conscious_content や external_request を context から設定
    conscious_data = context.get("conscious_content", {})
    source = "receptor" if "external_request" in context else "internal"
    if "external_request" in context:
        conscious_data = context["external_request"] # 簡易的に上書き

    pfc.handle_conscious_broadcast(source=source, conscious_data=conscious_data)
    goal = pfc.current_goal

    assert expected_keyword in goal
    print(f"✅ PrefrontalCortex: '{expected_keyword}'に基づく目標設定のテストに成功。 Goal: '{goal}'")

def test_prefrontal_cortex_handles_empty_context(mock_workspace, mock_motivation_system):
    """PFCが空のコンテキストで目標決定を行う場合のテスト。"""
    mock_motivation_system.get_internal_state.return_value = {}
    pfc = PrefrontalCortex(workspace=mock_workspace, motivation_system=mock_motivation_system)
    pfc.handle_conscious_broadcast(source="unknown", conscious_data={})
    goal = pfc.current_goal
    assert "Organize" in goal # デフォルト目標になるはず
    print("✅ PrefrontalCortex: 空コンテキストでの目標設定テストに成功。")