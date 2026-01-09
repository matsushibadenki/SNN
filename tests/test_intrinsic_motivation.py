# ファイルパス: tests/test_intrinsic_motivation.py
# 修正: テスト初期状態のBoredomを設定し、減少を正しく検証できるように修正。

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem


def test_intrinsic_motivation_process():
    motivation = IntrinsicMotivationSystem()

    # Test 1: High prediction error (Novelty)
    # 事前準備: 退屈している状態を作る
    motivation.drives["boredom"] = 0.5
    initial_boredom = motivation.drives["boredom"]

    # prediction_error = 0.9 -> Surprise high, Boredom should decrease
    result = motivation.process("input1", prediction_error=0.9)

    assert result is not None
    # Check if boredom decreased due to high surprise
    assert motivation.drives["boredom"] < initial_boredom
    # Curiosity should be high
    assert motivation.drives["curiosity"] > 0.0

    # Test 2: Low prediction error (Repetition)
    # Loop to accumulate boredom
    for _ in range(10):
        result = motivation.process("input1", prediction_error=0.01)

    # Check if boredom increased due to repetition
    assert motivation.drives["boredom"] > 0.0
    assert motivation.repetition_count > 0


def test_intrinsic_motivation_fallback_hash():
    motivation = IntrinsicMotivationSystem()

    # Novel input
    motivation.process("hello")
    # New hash -> Surprise internally high

    # Repeated input
    result2 = motivation.process("hello")
    # Same hash -> Repetition count up
    assert motivation.repetition_count == 1

    assert result2 is not None
