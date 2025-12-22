# ファイルパス: scripts/run_all_tests.py
# 日本語タイトル: Master Test Runner (v20.1 集約版)
# 目的・内容:
#   プロジェクト内の全テストスイート（単体・統合テスト）を一括実行する。
#   特に v20.1 で導入された蒸留、睡眠、メタ認知、BitSpike 関連の検証を優先。

import unittest
import sys
import os
import logging

# テスト実行時のノイズを減らすためログレベルを調整
logging.basicConfig(level=logging.ERROR)

def run_tests():
    # 1. プロジェクトルートをパスに追加 (インポートエラー防止)
    # scripts/run_all_tests.py から見たプロジェクトルート
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 環境変数 PYTHONPATH にも追加 (サブプロセス実行時用)
    os.environ["PYTHONPATH"] = project_root + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

    # 2. テストのセットアップ
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    tests_dir = os.path.join(project_root, "tests")
    
    print("====================================================")
    print("   Matsushiba SNN - Master Test Suite (v20.1)   ")
    print("====================================================")
    print(f"📂 Root: {project_root}")
    print(f"🧪 Discovering tests in: {tests_dir} ...")

    # 3. テストのディスカバリーと追加 (優先順位順)
    
    # フェーズ 20.1: 最重要・新機能テスト
    # - BitSpikeMamba, ThoughtDistiller, SleepConsolidator
    patterns = [
        "test_bit_spike_mamba.py",      # 1.58bit 量子化モデル
        "test_artificial_brain.py",     # 不確実性推定・サイクルロジック
        "test_cognitive_components.py", # 睡眠、蒸留、メタ認知
        "test_async_brain_kernel.py",   # 非同期イベント駆動
        "test_brain_integration.py",    # 統合シナリオ
    ]
    
    # 特定のパターンにマッチするテストを再帰的に検索
    for pattern in patterns:
        discovered_tests = loader.discover(start_dir=tests_dir, pattern=pattern)
        suite.addTests(discovered_tests)
        
    # 上記に含まれない汎用的なスモークテストの追加
    suite.addTests(loader.discover(start_dir=tests_dir, pattern="test_smoke_*.py"))

    # 4. 実行
    print(f"🚀 Running {suite.countTestCases()} test cases...\n")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 5. 結果レポート
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED! The Integrated Brain is healthy.")
        print("="*50)
        sys.exit(0)
    else:
        print(f"❌ FAILED: {len(result.failures)} failures, {len(result.errors)} errors detected.")
        print("Please review the detailed logs above.")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
