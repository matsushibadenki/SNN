# ファイルパス: tests/test_visual_cortex.py
# 日本語タイトル: 視覚野モデル (VisualCortex) 単体テスト v2.1
# 目的・内容: 
#   視覚野のBitNet動作および時系列処理の検証。
#   修正: リセットテスト時に model.eval() を適用し、ノイズの影響を排除。

import unittest
import torch
from snn_research.models.bio.visual_cortex import VisualCortex

class TestVisualCortex(unittest.TestCase):
    def setUp(self):
        # テスト用の設定
        self.in_channels = 3
        self.base_channels = 16 # テスト用に小さく設定
        self.time_steps = 5
        self.neuron_params = {"tau_mem": 20.0, "base_threshold": 1.0}

    def test_visual_cortex_static_image(self):
        """静止画入力に対する視覚野の動作テスト"""
        model = VisualCortex(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            time_steps=self.time_steps,
            neuron_params=self.neuron_params
        )
        
        # (Batch, Channel, H, W)
        x = torch.randn(2, 3, 32, 32)
        
        # モデルは (B, T, Features) を返す
        output = model(x)
        
        # 出力形状の確認
        self.assertEqual(output.dim(), 3)
        self.assertEqual(output.shape[0], 2) # Batch
        self.assertEqual(output.shape[1], self.time_steps) # Time
        # Output Dim = base_channels * 8 (IT layer)
        self.assertEqual(output.shape[2], self.base_channels * 8)

    def test_visual_cortex_video_stream(self):
        """動画ストリーム（時系列画像）に対する動作テスト"""
        model = VisualCortex(
            in_channels=1, # モノクロ動画
            base_channels=self.base_channels,
            time_steps=self.time_steps, # 動画の長さが優先されるが、デフォルト値として渡す
            neuron_params=self.neuron_params
        )
        
        # (Batch, Time, Channel, H, W)
        # Time=8 (モデルのデフォルト5より長い入力)
        x = torch.randn(2, 8, 1, 32, 32)
        
        output = model(x)
        
        # 出力の時間次元が入力と一致することを確認
        self.assertEqual(output.shape[1], 8)
        self.assertEqual(output.shape[0], 2)

    def test_reset(self):
        """内部状態のリセット機能のテスト"""
        model = VisualCortex(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            time_steps=self.time_steps,
            neuron_params=self.neuron_params
        )
        
        # [修正] 決定論的な動作を保証するために eval モードにする
        # これにより、Dropoutやニューロンのノイズ注入が無効化される
        model.eval()
        
        x = torch.randn(1, 3, 16, 16)
        
        # 1回目の実行
        out1 = model(x)
        
        # リセットを手動で呼ぶ
        model.reset_state()
        
        # 2回目の実行
        out2 = model(x)
        
        # 出力が厳密に一致することを確認
        self.assertTrue(torch.allclose(out1, out2), "Output mismatch after reset in eval mode")

if __name__ == "__main__":
    unittest.main()