# scripts/tests/run_all_tests.py
# ディレクトリ: scripts/tests
# 日本語タイトル: 全テスト実行ランナー
# 説明: プロジェクト内のすべてのpytestテストを一括実行する。
#       新しく追加された感情・身体性モデルのテストも対象となる。

import subprocess
import sys
import os
import time

def run_command(command, description):
    print(f"\n>>> Running: {description} ...")
    start_time = time.time()
    result = subprocess.call(command, shell=True)
    end_time = time.time()
    
    if result == 0:
        print(f"✅ {description} Passed ({end_time - start_time:.2f}s)")
        return True
    else:
        print(f"❌ {description} Failed")
        return False

def main():
    print("========================================")
    print("   SNN Research Project - Test Suite    ")
    print("========================================")
    
    # プロジェクトルートへのパス
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)
    
    # 1. ヘルスチェック
    if not run_command("python scripts/tests/run_project_health_check.py", "Project Health Check"):
        print("Health check failed. Aborting tests.")
        sys.exit(1)

    # 2. Pytest実行
    # tests/ ディレクトリ以下をすべて再帰的に探索して実行する
    # 新規作成した tests/models/test_emotional_brain.py 等も自動的に含まれる
    print("\n>>> Running All Unit Tests (pytest) ...")
    pytest_cmd = "python -m pytest tests/ -v"
    
    if run_command(pytest_cmd, "Unit Tests"):
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()