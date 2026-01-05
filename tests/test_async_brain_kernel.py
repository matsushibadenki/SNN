# ファイルパス: tests/test_async_brain_kernel.py
# 日本語タイトル: Async Brain Kernel Unit Tests
# 目的・内容:
#   AsyncEventBusとAsyncArtificialBrainの動作検証。
#   非同期メッセージングとイベントループの挙動を確認する。

import unittest
import asyncio
import sys
import os

# プロジェクトルートへのパス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain, AsyncEventBus, BrainEvent
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

class TestAsyncEventBus(unittest.TestCase):
    def test_pub_sub(self):
        """イベントの発行と購読のテスト"""
        bus = AsyncEventBus()
        received_events = []

        async def subscriber(event: BrainEvent):
            received_events.append(event)

        async def run_test():
            bus.subscribe("TEST_EVENT", subscriber)
            
            # ワーカー起動
            task = asyncio.create_task(bus.dispatch_worker())
            
            # イベント発行
            await bus.publish(BrainEvent("TEST_EVENT", "test_source", "payload_data"))
            
            # 処理待ち
            await asyncio.sleep(0.1)
            task.cancel()
            
        asyncio.run(run_test())
        
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload, "payload_data")

class TestAsyncBrainKernel(unittest.TestCase):
    def setUp(self):
        self.astrocyte = AstrocyteNetwork()
        
    def test_module_execution(self):
        """モジュール実行のテスト"""
        
        class MockModule:
            def forward(self, x):
                return f"processed_{x}"

        brain = AsyncArtificialBrain(
            modules={"test_module": MockModule()},
            astrocyte=self.astrocyte
        )

        async def run_kernel_test():
            await brain.start()
            
            # イベントバス経由ではなく、内部メソッドを直接テスト
            result_event = None
            
            # イベントリスナーをモック
            async def result_catcher(event):
                nonlocal result_event
                result_event = event

            brain.bus.subscribe("OUTPUT_EVENT", result_catcher)
            
            # モジュール実行
            await brain._run_module("test_module", "input", "OUTPUT_EVENT")
            
            await asyncio.sleep(0.1)
            await brain.stop()
            return result_event

        result = asyncio.run(run_kernel_test())
        
        self.assertIsNotNone(result)
        self.assertEqual(result.payload, "processed_input")

if __name__ == "__main__":
    unittest.main()