# ファイルパス: tests/test_universal_encoder.py
# 日本語タイトル: Universal Spike Encoder 機能検証
# 目的・内容:
#   Phase 8-6: 各モダリティ（Image, Audio, Text, DVS）が
#   正しく (Batch, Time, Features) のスパイク列に変換されるか検証。

import torch
import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.io.universal_encoder import UniversalSpikeEncoder

class TestUniversalEncoder(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.time_steps = 10
        self.device = 'cpu'
        self.encoder = UniversalSpikeEncoder(time_steps=self.time_steps, device=self.device)

    def test_image_rate_coding(self):
        """画像 (B, C, H, W) -> Rate Coding -> (B, T, F)"""
        print("\n[Test] Image Rate Coding")
        C, H, W = 3, 32, 32
        image = torch.rand(self.batch_size, C, H, W) # 0-1
        
        spikes = self.encoder.encode(image, modality='image', method='rate')
        
        print(f"Input shape: {image.shape}")
        print(f"Output shape: {spikes.shape}")
        
        expected_features = C * H * W
        self.assertEqual(spikes.shape, (self.batch_size, self.time_steps, expected_features))
        self.assertTrue(torch.all((spikes == 0.0) | (spikes == 1.0)))

    def test_image_latency_coding(self):
        """画像 (B, F) -> Latency Coding -> (B, T, F)"""
        print("\n[Test] Image Latency Coding")
        features = 100
        image_flat = torch.rand(self.batch_size, features)
        
        spikes = self.encoder.encode(image_flat, modality='image', method='latency')
        
        # Latency Codingでは、各特徴量につき1回だけ発火するはず (この簡易実装では)
        fire_counts = spikes.sum(dim=1) # (B, F)
        
        # 0.0-1.0のランダム入力なので、ほぼ確実に1回発火するが、
        # 端点処理等でズレる可能性もあるので、ここでは「多重発火していない」ことを確認
        self.assertTrue(torch.all(fire_counts <= 1.0))
        self.assertEqual(spikes.shape, (self.batch_size, self.time_steps, features))

    def test_audio_delta_coding(self):
        """音声 (B, T_in, F) -> Delta Coding -> (B, T_out, F)"""
        print("\n[Test] Audio Delta Coding")
        input_time = 20
        features = 64
        waveform = torch.sin(torch.linspace(0, 10, input_time)).unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, features)
        
        spikes = self.encoder.encode(waveform, modality='audio', method='delta')
        
        self.assertEqual(spikes.shape, (self.batch_size, self.time_steps, features))
        self.assertTrue(torch.all((spikes == 0.0) | (spikes == 1.0)))

    def test_text_embedding_coding(self):
        """テキストEmbedding (B, EmbedDim) -> Rate Coding -> (B, T, F)"""
        print("\n[Test] Text Embedding Coding")
        embed_dim = 256
        embedding = torch.randn(self.batch_size, embed_dim) # -inf ~ inf
        
        spikes = self.encoder.encode(embedding, modality='text', method='rate')
        
        self.assertEqual(spikes.shape, (self.batch_size, self.time_steps, embed_dim))
        # Sigmoidを通しているため発火することを確認
        self.assertGreater(spikes.sum(), 0)

    def test_dvs_passthrough(self):
        """DVS (B, T, C, H, W) -> Flatten -> (B, T, F)"""
        print("\n[Test] DVS Pass-through")
        # 入力Tと出力Tが異なる場合の調整もテスト
        input_time = 15 
        C, H, W = 2, 34, 34
        dvs_data = (torch.rand(self.batch_size, input_time, C, H, W) > 0.9).float()
        
        spikes = self.encoder.encode(dvs_data, modality='dvs')
        
        expected_features = C * H * W
        # Encoder設定の time_steps (10) にリサイズされているはず (Padding or Crop)
        self.assertEqual(spikes.shape, (self.batch_size, self.time_steps, expected_features))

if __name__ == '__main__':
    unittest.main()