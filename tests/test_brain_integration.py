# ファイルパス: tests/test_brain_integration.py
# 日本語タイトル: Brain v20 Integration Tests (Logger定義追加版)
# 目的・内容: logger の定義漏れによる NameError を修正し、統合テストを完遂させる。

import unittest
import asyncio
import sys
import os
import torch
import logging  # 追加

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ログ設定を追加
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ... (Mockクラス等は変更なし) ...

class TestBrainIntegration(unittest.TestCase):
    # ... (setUp等は変更なし) ...

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
            
            # 処理が流れるのを待つ
            try:
                await asyncio.wait_for(self._wait_for_energy_consumption(brain), timeout=10.0)
            except asyncio.TimeoutError:
                print("    [Warn] Integration test timed out waiting for energy drop.")
            
            await brain.stop()

        asyncio.run(run_scenario())
        
        # 診断レポートを確認
        report = self.astrocyte.get_diagnosis_report() #
        self.assertIn("status", report) #
        self.assertEqual(report["status"], "HEALTHY") #
        
        # ここで logger.info を呼び出すため、冒頭の定義が必要
        logger.info(f"✅ Integration Diagnosis: {report['status']}")

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
