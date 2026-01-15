# tests/test_bit_spike_mamba.py
# 修正: インポート先を models.transformer から models.experimental に変更

import pytest
import torch
import sys
import os

# プロジェクトルートをパスに追加（念のため）
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# [修正] 正しいパスからインポート
from snn_research.models.experimental.bit_spike_mamba import (
    BitLinear, bit_quantize_weight, BitSpikeMambaModel
)

class TestBitSpikeComponents:
    def test_linear_layer_forward(self):
        layer = BitLinear(in_features=16, out_features=32)
        x = torch.randn(2, 16)
        y = layer(x)
        assert y.shape == (2, 32)

    def test_weight_quantization(self):
        """BitSpikeの重み量子化ロジックのテスト"""
        weight = torch.randn(10, 10)

        # 戻り値の数をチェックしてアンパック
        outputs = bit_quantize_weight(weight)
        
        # 戻り値が (quantized, scale) のタプルであることを想定
        if isinstance(outputs, tuple):
            quantized = outputs[0]
        else:
            quantized = outputs
            
        assert quantized.shape == weight.shape
        # 値が -1, 0, 1 に近いかチェック (BitNet 1.58bit)
        unique_vals = torch.unique(quantized)
        assert len(unique_vals) <= 3 

class TestBitSpikeMambaModel:
    def test_model_forward(self):
        model = BitSpikeMambaModel(
            dim=64, depth=2, vocab_size=100
        )
        x = torch.randint(0, 100, (1, 10)) # (Batch, Seq)
        y = model(x)
        assert y.shape == (1, 10, 100) # (Batch, Seq, Vocab)

    def test_model_size_calculation(self):
        model = BitSpikeMambaModel(dim=64, depth=2, vocab_size=100)
        # エラーが出なければOK
        size_info = model.print_model_size()
        # printするだけのメソッドならNoneが返る場合が多い