# ファイルパス: tests/test_liquid_association.py
# 日本語タイトル: Liquid Association Cortex 統合テスト
# 目的・内容:
#   Phase 9-9: Universal Spike Encoder と LAC の結合テスト。
#   異なるモダリティの入力をエンコードし、リザーバに時系列入力して、
#   ニューロン活動が起きるか（＝統合されているか）を確認する。

import torch
import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.io.universal_encoder import UniversalSpikeEncoder
from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex

class TestLiquidAssociation(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.time_steps = 10
        self.device = 'cpu'
        
        # Encoder初期化
        self.encoder = UniversalSpikeEncoder(time_steps=self.time_steps, device=self.device)
        
        # 入力次元定義
        self.dim_visual = 32 * 32 * 3 # Image flatten
        self.dim_audio = 64           # Audio features
        self.dim_text = 128           # Text embedding
        self.dim_somato = 10          # Somato sensors
        self.reservoir_size = 500
        
        # LAC初期化
        self.lac = LiquidAssociationCortex(
            num_visual_inputs=self.dim_visual,
            num_audio_inputs=self.dim_audio,
            num_text_inputs=self.dim_text,
            num_somato_inputs=self.dim_somato,
            reservoir_size=self.reservoir_size
        ).to(self.device)

    def test_multimodal_integration(self):
        """3つのモダリティ（視覚・聴覚・言語）を同時に入力して統合を確認"""
        print("\n[Test] LAC Multimodal Integration")
        
        # 1. ダミーデータの生成
        raw_image = torch.rand(self.batch_size, 3, 32, 32)
        raw_audio = torch.rand(self.batch_size, 20, self.dim_audio) # (B, Time, Feat)
        raw_text = torch.randn(self.batch_size, self.dim_text)      # Embedding
        
        # 2. スパイクエンコード (Batch, Time, Features)
        spike_visual = self.encoder.encode(raw_image, 'image', 'rate')
        spike_audio = self.encoder.encode(raw_audio, 'audio', 'delta')
        spike_text = self.encoder.encode(raw_text, 'text', 'rate')
        
        # 3. LACへの時系列入力
        self.lac.reset_state()
        total_activity = 0.0
        
        print(f"Feeding {self.time_steps} steps into Reservoir...")
        
        for t in range(self.time_steps):
            # 各ステップのスパイクを取り出す (Batch, Features)
            v_in = spike_visual[:, t, :]
            a_in = spike_audio[:, t, :]
            t_in = spike_text[:, t, :]
            
            # LAC Forward (SomatoはNone)
            reservoir_out = self.lac(
                visual_spikes=v_in,
                audio_spikes=a_in,
                text_spikes=t_in,
                somato_spikes=None
            )
            
            step_activity = reservoir_out.sum().item()
            total_activity += step_activity
            # print(f"  Step {t}: Active Neurons = {step_activity}")
            
        print(f"Total Reservoir Activity: {total_activity}")
        
        # アサーション: リザーバが何らかの応答をしていること
        self.assertGreater(total_activity, 0, "Reservoir remained silent despite multimodal input.")
        
        # 形状確認
        self.assertEqual(reservoir_out.shape, (self.batch_size, self.reservoir_size))

    def test_single_modality(self):
        """単一モダリティ（視覚のみ）でも動作することを確認"""
        print("\n[Test] LAC Single Modality (Visual Only)")
        
        raw_image = torch.rand(self.batch_size, 3, 32, 32)
        spike_visual = self.encoder.encode(raw_image, 'image', 'rate')
        
        self.lac.reset_state()
        total_activity = 0.0
        
        for t in range(self.time_steps):
            v_in = spike_visual[:, t, :]
            # 他はNone
            out = self.lac(visual_spikes=v_in)
            total_activity += out.sum().item()
            
        print(f"Total Visual-only Activity: {total_activity}")
        self.assertGreater(total_activity, 0)

if __name__ == '__main__':
    unittest.main()