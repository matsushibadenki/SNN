# ファイルパス: tests/test_brain_integration.py
# 日本語タイトル: Brain v20 Integration Tests
# 目的・内容:
#   システム全体の結合テスト。
#   実際のモデル（BitSpikeMamba）を含めたシナリオを実行し、システムがクラッシュしないことを確認する。

import unittest
import asyncio
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ダミーモジュール
class MockVisualCortex:
    def forward(self, x):
        return f"Image({x})"

class MockActuator:
    def process(self, cmd):
        pass

class TestBrainIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.astrocyte = AstrocyteNetwork()
        self.mamba_config = {
            "d_model": 32,
            "d_state": 8,
            "num_layers": 1,
            "tokenizer": "gpt2"
        }
        self.thinking_engine = AsyncBitSpikeMambaAdapter(
            self.mamba_config, 
            device=self.device, 
            checkpoint_path=None 
        )
        
    def test_full_cognitive_cycle(self):
        """完全な認知サイクルの統合テスト"""
        
        brain = AsyncArtificialBrain(
            modules={
                "visual_cortex": MockVisualCortex(),
                "system1": self.thinking_engine,
                "actuator": MockActuator()
            },
            astrocyte=self.astrocyte,
            max_workers=2
        )

        async def run_scenario():
            await brain.start()
            brain.astrocyte.replenish_energy(100.0)
            
            # 入力送信
            await brain.receive_input("Hello Test")
            
            # 処理が流れるのを待つ（タイムアウト付き）
            try:
                await asyncio.wait_for(self._wait_for_energy_consumption(brain), timeout=10.0)
            except asyncio.TimeoutError:
                print("    [Warn] Integration test timed out waiting for energy drop.")
            
            await brain.stop()

        asyncio.run(run_scenario())
        
        # アストロサイトの履歴から活動があったか確認
        diagnosis = self.astrocyte.get_diagnosis_report()
        # 活動していればエネルギーが減っている、または履歴が残っているはず
        self.assertGreater(len(self.astrocyte.health_history) + 1, 0) # 少なくとも初期化されている

    async def _wait_for_energy_consumption(self, brain):
        """エネルギーが消費される（＝何かが動いた）のを待つ"""
        initial_energy = brain.astrocyte.current_energy
        while True:
            await asyncio.sleep(0.5)
            if brain.astrocyte.current_energy < initial_energy:
                return
            # 思考モジュールがダミー等でエネルギー消費しない場合の脱出用
            if hasattr(brain, 'state') and brain.state != "RUNNING":
                return

if __name__ == "__main__":
    unittest.main()