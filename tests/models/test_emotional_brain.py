# tests/models/test_emotional_brain.py
# ディレクトリ: tests/models
# 日本語タイトル: 感情概念脳の単体テスト
# 説明: EmotionalConceptBrainとAmygdalaの動作検証を行う。

import pytest
import torch
from snn_research.models.hybrid.emotional_concept_brain import EmotionalConceptBrain
from snn_research.cognitive_architecture.amygdala import Amygdala

def test_amygdala_initialization():
    """Amygdalaの初期化と出力範囲のテスト"""
    amygdala = Amygdala(input_dim=10, hidden_dim=5)
    dummy_input = torch.randn(2, 10)
    output = amygdala(dummy_input)
    
    assert output.shape == (2, 1), "Amygdala output shape should be (Batch, 1)"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Value should be tanh range (-1 to 1)"

def test_emotional_brain_forward():
    """EmotionalConceptBrainの順伝播テスト"""
    brain = EmotionalConceptBrain(num_classes=10)
    dummy_img = torch.randn(2, 1, 28, 28) # Batch=2
    
    logits, emotion = brain(dummy_img)
    
    assert logits.shape == (2, 10), "Logits shape mismatch"
    assert emotion.shape == (2, 1), "Emotion value shape mismatch"
    
    # 内部状態が保持されているか確認
    internal_state = brain.get_internal_state()
    assert internal_state is not None
    assert internal_state.shape == (2, 128)

def test_emotional_brain_consistency():
    """同じ入力に対して決定論的に同じ感情を返すか（学習前）"""
    brain = EmotionalConceptBrain()
    brain.eval() # 評価モード（Dropout等を無効化）
    img = torch.randn(1, 1, 28, 28)
    
    logits1, emo1 = brain(img)
    logits2, emo2 = brain(img)
    
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(emo1, emo2)