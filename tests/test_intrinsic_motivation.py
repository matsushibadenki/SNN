from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem


def test_intrinsic_motivation_process():
    motivation = IntrinsicMotivationSystem()

    # Test 1: High prediction error (Novelty)
    # prediction_error = 0.9 -> Surprise should be high (~0.9), Boredom should decrease
    result = motivation.process("input1", prediction_error=0.9)
    assert result is not None
    assert result["surprise"] == 0.9
    assert motivation.drives["boredom"] < 0.1  # Should drop significantly

    # Test 2: Low prediction error (Repetition)
    # prediction_error = 0.01 -> Surprise low, Boredom should increase
    # Loop to accumulate boredom
    for _ in range(10):
        result = motivation.process("input1", prediction_error=0.01)

    assert result["surprise"] == 0.01
    assert motivation.drives["boredom"] > 0.0  # Should be increasing
    assert motivation.repetition_count > 0


def test_intrinsic_motivation_fallback_hash():
    motivation = IntrinsicMotivationSystem()

    # Novel input
    result1 = motivation.process("hello")
    assert result1["surprise"] == 1.0  # New hash

    # Repeated input
    result2 = motivation.process("hello")
    assert result2["surprise"] == 0.0  # Same hash
    assert motivation.repetition_count == 1
