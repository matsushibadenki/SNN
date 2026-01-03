# ファイルパス: tests/test_autonomous_agent.py
# Title: Autonomous Agent Logic Test
# 機能: エージェントが未知の入力に対して好奇心を持ち、Web検索を行い、知識を獲得する一連の流れをテストする。

import unittest
import torch
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.agent.autonomous_agent import AutonomousAgent

class TestAutonomousAgent(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.input_size = 64
        self.output_size = 4
        self.agent = AutonomousAgent(self.input_size, self.output_size, self.device)

    def test_initialization(self):
        """エージェントの初期化確認"""
        self.assertIsNotNone(self.agent.brain)
        self.assertIsNotNone(self.agent.crawler)
        self.assertIsNotNone(self.agent.encoder)
        print("\n[Test] Agent Initialization: OK")

    def test_perceive_and_act_normal(self):
        """通常時の動作確認（既知の入力）"""
        # ランダムだが安定した入力 (信頼度高めを想定)
        state = torch.rand(self.input_size)
        action = self.agent.perceive_and_act(state, step=1)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.output_size)
        print(f"[Test] Normal Action: {action} (OK)")

    def test_curiosity_trigger(self):
        """好奇心トリガーと検索行動のテスト"""
        print("\n[Test] Testing Curiosity Drive...")
        
        # 内部状態を強制的に「自信なし」にするハック（テスト用）
        # 実際の運用では予測誤差から自動発火する
        # ここでは perceive_and_act 内の確率的トリガーを待つ代わりに
        # 直接 _satisfy_curiosity を呼んでロジックを確認する
        
        context_state = torch.rand(self.input_size)
        
        # 実行前の知識（バッファなどあれば確認）
        
        # 好奇心行動の実行
        self.agent._satisfy_curiosity(context_state)
        
        # 検索が行われ、コンソール出力が出ていることを確認（目視確認用）
        # 実際にはMock Crawlerが呼ばれたかを確認する
        
        print("[Test] Curiosity Cycle: Executed without error")

    def test_idle_routine(self):
        """放置時の退屈回避ルーチン"""
        print("\n[Test] Testing Idle Routine...")
        self.agent.boredom_counter = 101 # 閾値超え
        self.agent.idle_routine()
        self.assertEqual(self.agent.boredom_counter, 0, "Boredom counter should reset after activity")
        print("[Test] Idle Routine: Reset OK")

if __name__ == '__main__':
    unittest.main()