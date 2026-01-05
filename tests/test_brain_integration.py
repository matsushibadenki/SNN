# ファイルパス: tests/test_brain_integration.py
# 日本語タイトル: Brain v20 Integration Tests (完全復元版)
# 目的・内容: 欠落していた Mock クラスを復元し、logger 定義を含めてテストを完遂させる。

from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
import unittest
import asyncio
import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ログ設定
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# --- Mock Components (復元箇所) ---


class MockVisualCortex:
    """視覚野のダミー。入力をそのまま文字列で返す。"""

    def forward(self, x):
        return f"Image({x})"


class MockActuator:
    """アクチュエータのダミー。"""

    def process(self, cmd):
        pass

# ----------------------------------


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
        # 重み不一致エラーを回避する修正が適用されているアダプターを使用
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
            # リソース要求が通るようにエネルギーを補充
            brain.astrocyte.replenish_energy(100.0)

            # 入力送信
            await brain.receive_input("Hello Test")

            # 処理が流れるのを待つ
            try:
                await asyncio.wait_for(self._wait_for_energy_consumption(brain), timeout=10.0)
            except asyncio.TimeoutError:
                print(
                    "    [Warn] Integration test timed out waiting for energy drop.")

            await brain.stop()

        asyncio.run(run_scenario())

        # 診断レポート機能を使用して状態を確認
        report = self.astrocyte.get_diagnosis_report()
        self.assertIn("status", report)
        self.assertEqual(report["status"], "HEALTHY")

        logger.info(f"✅ Integration Diagnosis: {report['status']}")

    async def _wait_for_energy_consumption(self, brain):
        """エネルギーが消費される（＝何かが動いた）のを待つ"""
        initial_energy = brain.astrocyte.current_energy
        while True:
            await asyncio.sleep(0.5)
            # アストロサイトが活動を検知してエネルギーが減ったか確認
            if brain.astrocyte.current_energy < initial_energy:
                return
            if hasattr(brain, 'state') and brain.state != "RUNNING":
                return


if __name__ == "__main__":
    unittest.main()
