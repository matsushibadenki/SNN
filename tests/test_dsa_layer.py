# ファイルパス: tests/test_dsa_layer.py
# 日本語タイトル: SNN-DSAレイヤー機能テスト [Fixed]
# 目的・内容:
#   DynamicSparseAttention クラスの単体テスト。
#   修正点: 閾値を下げて発火を確認するアサーションを追加。

import torch
import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.layers.dsa import DynamicSparseAttention

class TestDynamicSparseAttention(unittest.TestCase):
    
    def setUp(self):
        # テストパラメータ設定
        self.batch_size = 2
        self.time_steps = 10
        self.d_model = 32
        self.num_heads = 4
        self.top_k = 3
        
        self.device = torch.device("cpu")
        
        # テスト用に閾値を低く設定して発火しやすくする
        neuron_params = {'base_threshold': 0.1}
        
        self.dsa = DynamicSparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            top_k=self.top_k,
            neuron_params=neuron_params
        ).to(self.device)
        
        # ダミー入力スパイク (密度高め)
        self.input_spikes = (torch.rand(self.batch_size, self.time_steps, self.d_model) > 0.5).float().to(self.device)

    def test_forward_shape(self):
        """順伝播の出力形状が入力と一致するか確認"""
        print("\n[Test] Forward Shape Check")
        out_spikes, attn_maps = self.dsa(self.input_spikes)
        
        print(f"Input shape: {self.input_spikes.shape}")
        print(f"Output shape: {out_spikes.shape}")
        
        self.assertEqual(out_spikes.shape, self.input_spikes.shape)
        expected_attn_shape = (self.batch_size, self.num_heads, self.time_steps, self.time_steps)
        self.assertEqual(attn_maps.shape, expected_attn_shape)

    def test_sparsity_top_k(self):
        """Top-K スパース性が正しく適用されているか確認"""
        print("\n[Test] Sparsity Top-K Check")
        _, attn_maps = self.dsa(self.input_spikes)
        
        # 閾値を設けて非ゼロ要素をカウント
        non_zero_count = (attn_maps > 1e-9).sum(dim=-1)
        
        print(f"Max active keys per query: {non_zero_count.max().item()}")
        print(f"Target Top-K: {self.top_k}")
        
        self.assertTrue(torch.all(non_zero_count <= self.top_k))

    def test_spike_output(self):
        """出力がスパイク（0/1）であり、かつ発火しているか確認"""
        print("\n[Test] Spike Output Check")
        out_spikes, _ = self.dsa(self.input_spikes)
        
        unique_vals = torch.unique(out_spikes)
        print(f"Unique output values: {unique_vals}")
        
        # 0と1のみが含まれているか
        is_binary = torch.all( (out_spikes == 0.0) | (out_spikes == 1.0) )
        self.assertTrue(is_binary, "Output should be binary (0. or 1.)")
        
        # 少なくとも1回は発火していることを確認 (Dead Neuron防止)
        total_spikes = out_spikes.sum().item()
        print(f"Total spikes generated: {total_spikes}")
        self.assertGreater(total_spikes, 0, "Neurons did not fire at all! Check threshold or normalization.")

if __name__ == '__main__':
    unittest.main()