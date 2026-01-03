# ファイルパス: scripts/tests/run_all_tests.py
# 日本語タイトル: マスターテストランナー (v20.5 統合版)
# 目的: プロジェクト内の全ユニットテストを一括実行する。

import unittest
import sys
import os

# プロジェクトルートの設定
# scripts/tests/run_all_tests.py から見て ../../ がルート
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# パスに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_all_tests():
    """
    プロジェクト全体のテストを実行するランナー
    """
    # テストディレクトリの特定 (プロジェクトルート直下の tests ディレクトリ)
    start_dir = os.path.join(project_root, "tests")
    
    if not os.path.exists(start_dir):
        print(f"❌ Error: Test directory not found at: {start_dir}")
        sys.exit(1)

    print("====================================================")
    print("   Matsushiba SNN - Master Test Suite (v17.2)")
    print("====================================================")
    print(f"🧪 Discovering all tests in: {start_dir}")

    # テストの探索
    # top_level_dirを指定することで、プロジェクトルートからのimportを正しく解決します
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern="test_*.py", top_level_dir=project_root)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()