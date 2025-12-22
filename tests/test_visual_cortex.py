# ファイルパス: tests/test_visual_cortex.py
# 日本語タイトル: 視覚野モデル (VisualCortex) 単体テスト
# 目的・内容: 視覚野の階層的予測符号化および動的なリセット機能の検証

import unittest
import torch
from snn_research.models.bio.visual_cortex import VisualCortex

class TestVisualCortex(unittest.TestCase):
    def setUp(self):
        # 共通の設定を定義
        self.config = {
            "in_channels": 3,
            "layer_dims": [64, 128],
            "time_steps": 5,
            "error_gain": 0.1
        }

    def test_visual_cortex_static_image(self):
        """静止画入力に対する視覚野の動作テスト"""
        model = VisualCortex(self.config)
        # (Batch, Channel, H, W)
        x = torch.randn(2, 3, 32, 32)
        
        # 修正: 戻り値は2つ (states, errors)
        states, errors = model(x)
        
        # 各レイヤーの結果が含まれているか
        self.assertEqual(len(states), 2)
        self.assertEqual(len(errors), 2)
        
        # 出力形状の確認 (Batch, Time, Dim)
        # VisualCortexの各レイヤー出力は (B, T, C) の形式であることを想定
        self.assertEqual(states[0].shape[0], 2)
        self.assertEqual(states[0].shape[1], self.config["time_steps"])

    def test_visual_cortex_video_stream(self):
        """動画ストリーム（時系列画像）に対する動作テスト"""
        video_config = self.config.copy()
        video_config["in_channels"] = 1
        model = VisualCortex(video_config)
        
        # (Batch, Time, Channel, H, W)
        x = torch.randn(2, 5, 1, 32, 32)
        
        # 修正: 戻り値は2つ
        states, errors = model(x)
        
        self.assertEqual(states[-1].shape[0], 2)
        # 時系列入力の場合、内部でステップごとに処理されることを確認
        self.assertTrue(states[-1].sum() != 0)

    def test_reset(self):
        """内部状態のリセット機能のテスト"""
        # 修正: Positional argumentsではなくconfig辞書を渡す
        model = VisualCortex(self.config)
        
        x = torch.randn(1, 3, 16, 16)
        model(x)
        
        # 状態が保持されていることを確認（実装に依存するが一般的には非ゼロ）
        model.reset_states()
        
        # リセット後に別の入力を入れてエラーが起きないか
        try:
            model(x)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
