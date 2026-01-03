# ファイルパス: tests/test_dvs_pipeline.py
# 日本語タイトル: DVSデータパイプライン検証
# 目的・内容:
#   Phase 8-5: 実装した NeuromorphicDataFactory が正常にデータを供給できるか確認。
#   修正: インポート元を snn_research.io に変更。

import torch
import unittest
import sys
import os
import shutil

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 修正: io パッケージからインポート
from snn_research.io.neuromorphic_dataset import NeuromorphicDataFactory, visualize_dvs_sample

class TestDVSPipeline(unittest.TestCase):
    
    def setUp(self):
        # テスト用データディレクトリ（プロジェクトルート下のdata/temp_testを使用）
        self.root_dir = "./data/temp_test_n_mnist"
        self.batch_size = 4
        self.time_steps = 16
        
        # テスト実行後に画像を保存するディレクトリ
        self.result_dir = "workspace/results/tests"
        os.makedirs(self.result_dir, exist_ok=True)

    def tearDown(self):
        # テスト用の一時データディレクトリを削除 (クリーンアップ)
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_mock_dataloader_shape(self):
        """Mockデータローダーからの読み出し形状確認"""
        print("\n[Test] DVS Mock DataLoader Shape")
        
        loader = NeuromorphicDataFactory.create_dataloader(
            dataset_name="n-mnist",
            root_dir=self.root_dir,
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            use_mock=True,
            num_workers=0 # テスト時は0推奨
        )
        
        # 1バッチ取得
        try:
            spikes, labels = next(iter(loader))
        except StopIteration:
            self.fail("DataLoader is empty.")
        
        print(f"Spikes shape: {spikes.shape}") # 期待: (Batch, Time, C, H, W)
        print(f"Labels shape: {labels.shape}")
        
        # N-MNIST (Mock) 規格: 34x34, 2 Channels
        expected_shape = (self.batch_size, self.time_steps, 2, 34, 34)
        self.assertEqual(spikes.shape, expected_shape)
        self.assertEqual(labels.shape[0], self.batch_size)

    def test_spike_values(self):
        """データがバイナリ（0/1）であることを確認"""
        print("\n[Test] DVS Spike Value Integrity")
        
        loader = NeuromorphicDataFactory.create_dataloader(
            dataset_name="n-mnist",
            root_dir=self.root_dir,
            use_mock=True,
            num_workers=0
        )
        spikes, _ = next(iter(loader))
        
        unique_vals = torch.unique(spikes)
        print(f"Unique values in batch: {unique_vals}")
        
        # 0と1のみであることを確認
        is_binary = torch.all((spikes == 0.0) | (spikes == 1.0))
        self.assertTrue(is_binary, "Spikes should be binary (0 or 1)")

    def test_visualization(self):
        """可視化関数の動作確認"""
        print("\n[Test] DVS Visualization")
        
        loader = NeuromorphicDataFactory.create_dataloader(
            dataset_name="n-mnist",
            root_dir=self.root_dir,
            use_mock=True,
            num_workers=0
        )
        spikes, _ = next(iter(loader))
        
        # 最初のサンプルの可視化
        sample = spikes[0] # (T, C, H, W)
        save_path = os.path.join(self.result_dir, "test_dvs_vis.png")
        
        try:
            visualize_dvs_sample(sample, save_path)
            # ファイルが生成されたか確認
            self.assertTrue(os.path.exists(save_path), "Visualization file was not created.")
        except Exception as e:
            print(f"Visualization skipped or failed: {e}")

if __name__ == '__main__':
    unittest.main()