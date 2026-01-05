# ファイルパス: tests/test_bit_spike_mamba.py
# 日本語タイトル: BitSpikeMamba Unit Tests
# 目的・内容:
#   BitSpikeLinearおよびBitSpikeMambaモデルの機能検証。
#   量子化の正確性と推論パイプラインの整合性をチェックする。

import unittest
import torch
import sys
import os

# プロジェクトルートへのパス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snn_research.core.layers.bit_spike_layer import BitSpikeLinear, bit_quantize_weight
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

class TestBitSpikeComponents(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_weight_quantization(self):
        """BitSpikeの重み量子化ロジックのテスト"""
        # ランダムな重みを作成
        weight = torch.randn(10, 10)
        
        # 量子化実行
        quantized = bit_quantize_weight(weight)
        
        # 検証1: 値が離散化されているか（スケーリングを除く）
        # bit_quantize_weightは gamma * round(w/gamma) を返すため、
        # ユニークな値の数は多くなる可能性があるが、実質的には3値に近い分布になるはず。
        # ここでは簡易的に、BitSpikeLinearのforward内での挙動に近いチェックを行う。
        
        gamma = torch.mean(torch.abs(weight))
        scaled = quantized / gamma
        unique_vals = torch.unique(torch.round(scaled))
        
        # {-1, 0, 1} の範囲に含まれているか
        self.assertTrue(torch.all(unique_vals >= -1))
        self.assertTrue(torch.all(unique_vals <= 1))

    def test_linear_layer_forward(self):
        """BitSpikeLinear層のForwardパステスト"""
        layer = BitSpikeLinear(32, 16)
        input_tensor = torch.randn(1, 32) # バッチサイズ1
        output = layer(input_tensor)
        
        self.assertEqual(output.shape, (1, 16))
        self.assertFalse(torch.isnan(output).any())

class TestBitSpikeMambaModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 100,
            "d_model": 32,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 2,
            "time_steps": 2,
            "neuron_config": {"type": "lif", "tau_mem": 2.0}
        }
        self.model = BitSpikeMamba(**self.config)

    def test_model_forward(self):
        """モデル全体のForwardパステスト"""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        logits, spikes, mem = self.model(input_ids)
        
        # Logits形状確認: (B, L, Vocab)
        self.assertEqual(logits.shape, (batch_size, seq_len, 100))
        
        # スパイク出力確認
        self.assertEqual(spikes.shape, ()) # スカラー
        
    def test_model_size_calculation(self):
        """モデルサイズ計算機能のテスト"""
        size_mb = self.model.get_model_size_mb()
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0.0)
        print(f"    [Info] Estimated Model Size: {size_mb:.4f} MB")

if __name__ == "__main__":
    unittest.main()