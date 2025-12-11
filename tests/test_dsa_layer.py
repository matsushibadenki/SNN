# ファイルパス: tests/test_dsa_layer.py
# 日本語タイトル: SNN-DSAレイヤー機能テスト
# 目的・内容:
#   新規実装された DynamicSparseAttention クラスの単体テストを行う。
#   1. 順伝播の形状チェック
#   2. スパース性（Top-Kルーティング）の検証
#   3. スパイク出力の確認

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
        self.top_k = 3 # Time steps (10) より小さい値
        
        self.device = torch.device("cpu")
        
        self.dsa = DynamicSparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            top_k=self.top_k
        ).to(self.device)
        
        # ダミー入力スパイク (0 or 1)
        self.input_spikes = (torch.rand(self.batch_size, self.time_steps, self.d_model) > 0.8).float().to(self.device)

    def test_forward_shape(self):
        """順伝播の出力形状が入力と一致するか確認"""
        print("\n[Test] Forward Shape Check")
        out_spikes, attn_maps = self.dsa(self.input_spikes)
        
        print(f"Input shape: {self.input_spikes.shape}")
        print(f"Output shape: {out_spikes.shape}")
        
        self.assertEqual(out_spikes.shape, self.input_spikes.shape)
        # Attention Map shape: (B, Num_Heads, T, T)
        expected_attn_shape = (self.batch_size, self.num_heads, self.time_steps, self.time_steps)
        self.assertEqual(attn_maps.shape, expected_attn_shape)

    def test_sparsity_top_k(self):
        """Top-K スパース性が正しく適用されているか確認"""
        print("\n[Test] Sparsity Top-K Check")
        _, attn_maps = self.dsa(self.input_spikes)
        
        # attn_maps (Probabilities) の各行において、非ゼロ要素の数が top_k 以下であることを確認
        # Softmaxを通っているため、完全に0にはならない要素もあるが、
        # マスクされた部分は 0.0 (または非常に小さい値) になっているはず
        # ここでは実装上のマスク処理 (-inf -> softmax -> 0) を確認
        
        # 閾値を設けて非ゼロ要素をカウント
        non_zero_count = (attn_maps > 1e-9).sum(dim=-1)
        
        print(f"Max active keys per query: {non_zero_count.max().item()}")
        print(f"Target Top-K: {self.top_k}")
        
        # 全てのヘッド、全てのタイムステップ、全てのバッチで Top-K 以下であること
        self.assertTrue(torch.all(non_zero_count <= self.top_k))

    def test_spike_output(self):
        """出力がバイナリ（スパイク）に近いか確認"""
        print("\n[Test] Spike Output Check")
        out_spikes, _ = self.dsa(self.input_spikes)
        
        # SNNの出力は通常 0.0 か 1.0 (サロゲート勾配使用時は近似値になる場合もあるが、forwardでは0/1)
        unique_vals = torch.unique(out_spikes)
        print(f"Unique output values: {unique_vals}")
        
        # 0と1のみが含まれているか（浮動小数点誤差を考慮）
        is_binary = torch.all( (out_spikes == 0.0) | (out_spikes == 1.0) )
        # AdaptiveLIFの実装によってはサロゲート勾配の影響でforwardでも0/1以外が出る場合があるか確認
        # (通常 LearnableATan.forward は (input > 0).float() なので 0/1)
        self.assertTrue(is_binary)

if __name__ == '__main__':
    unittest.main()