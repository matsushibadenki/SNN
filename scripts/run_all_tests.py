# ファイルパス: scripts/run_all_tests.py
# 日本語タイトル: マスターテストランナー (v20.5 統合版)
# 目的・内容: プロジェクト内の全テストスイートを一括実行し、全機能の論理的整合性を検証する。

import unittest
import sys
import os
import logging

# ログレベル調整
logging.basicConfig(level=logging.WARNING)

def run_tests():
    # プロジェクトルートの設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = project_root + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

    loader = unittest.TestLoader()
    
    print("====================================================")
    print("   Matsushiba SNN - Master Test Suite (v20.5)   ")
    print("====================================================")
    print(f"🧪 Discovering all tests in: {os.path.join(project_root, 'tests')}")

    # testsディレクトリ内の全ての test_*.py を対象にする
    suite = loader.discover(start_dir=os.path.join(project_root, "tests"), pattern="test_*.py")

    print(f"🚀 Running {suite.countTestCases()} test cases...")
    
    # 実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("✅ SUCCESS: 全てのテストケースをクリアしました。")
        sys.exit(0)
    else:
        print(f"❌ FAILED: {len(result.failures)} 個の失敗, {len(result.errors)} 個のエラー")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
