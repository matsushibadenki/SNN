# ファイルパス: scripts/run_all_tests.py
# 日本語タイトル: Master Test Runner
# 目的・内容:
#   プロジェクト内の主要なテストスイートを一括実行し、レポートを表示する。

import unittest
import sys
import os
import logging

# ログレベルを上げてテスト出力をクリーンにする
logging.basicConfig(level=logging.ERROR)

def run_tests():
    # プロジェクトルートをパスに追加
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    
    # テストローダー
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # テストディレクトリ
    tests_dir = os.path.join(project_root, "tests")
    
    print("========================================")
    print("   Matsushiba SNN - Test Suite Runner   ")
    print("========================================")
    
    # テストのディスカバリーと追加
    # 1. BitSpikeMamba Tests
    suite.addTests(loader.discover(tests_dir, pattern="test_bit_spike_mamba.py"))
    
    # 2. Async Kernel Tests
    suite.addTests(loader.discover(tests_dir, pattern="test_async_brain_kernel.py"))
    
    # 3. Integration Tests
    suite.addTests(loader.discover(tests_dir, pattern="test_brain_integration.py"))
    
    # 実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! The Brain is healthy.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()