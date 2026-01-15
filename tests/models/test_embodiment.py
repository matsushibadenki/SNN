# tests/models/test_embodiment.py
# ディレクトリ: tests/models
# 日本語タイトル: 身体性エージェントの単体テスト
# 説明: EmotionalAgentとMotorCortexの連携動作を検証する。

import pytest
import torch
from snn_research.models.hybrid.emotional_concept_brain import EmotionalConceptBrain
from snn_research.models.embodied.emotional_agent import EmotionalAgent, MotorCortex

def test_motor_cortex_shape():
    """運動野の入出力シェイプテスト"""
    motor = MotorCortex(input_dim=129, hidden_dim=32, output_dim=2)
    cortex_state = torch.randn(4, 128)
    emotion = torch.randn(4, 1)
    
    action_logits = motor(cortex_state, emotion)
    assert action_logits.shape == (4, 2)

def test_agent_act_cycle():
    """エージェントの知覚-行動サイクルのテスト"""
    brain = EmotionalConceptBrain(num_classes=10)
    agent = EmotionalAgent(brain)
    
    img = torch.randn(1, 1, 28, 28)
    
    # 行動生成
    action_logits, emotion = agent.act(img)
    
    assert action_logits.shape == (1, 2)
    assert emotion.shape == (1, 1)
    
    # 感情値はBrainから来ているはず
    _, brain_emotion = brain(img)
    # act内で再計算している場合、evalモードでないとわずかにずれる可能性があるので近似比較
    assert torch.allclose(emotion, brain_emotion, atol=1e-5)