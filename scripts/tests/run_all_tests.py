# scripts/tests/run_all_tests.py
# ディレクトリ: scripts/tests
# 日本語タイトル: 全テスト実行ランナー
# 説明: プロジェクト内のすべてのpytestテストおよびscripts内の検証スクリプトを一括実行する。
#       scripts/tests内の個別テストや検証スクリプトも対象に含める。

import subprocess
import sys
import os
import time

def run_command(command, description):
    print(f"\n>>> Running: {description} ...")
    print(f"    Command: {command}")
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

    all_tests_passed = True

    # 2. Pytest実行 (tests/ ディレクトリ)
    # 標準的な単体テスト群
    print("\n>>> Running Standard Unit Tests (pytest tests/) ...")
    pytest_cmd = "python -m pytest tests/ -v"
    if not run_command(pytest_cmd, "Standard Unit Tests"):
        all_tests_passed = False

    # 3. 追加のスクリプトテスト実行 (scripts/tests/ ディレクトリ)
    # scripts/tests/ には test_*.py や run_compiler_test.py などが含まれているため
    # pytestでこのディレクトリも明示的にターゲットにするか、個別に実行する
    print("\n>>> Running Script Tests (pytest scripts/tests/) ...")
    # 注意: verify_*.py など test_ 接頭辞がないものもチェックする場合は設定が必要だが、
    # ここでは test_*.py と *_test.py を対象とする標準的なpytestを実行
    script_tests_cmd = "python -m pytest scripts/tests/ -v" 
    if not run_command(script_tests_cmd, "Script Tests"):
        all_tests_passed = False
    
    # 4. 主要な検証スクリプトの実行 (Verification Scripts)
    # Pytestで拾われない verify_*.py などを個別に実行して動作確認を行う
    verification_scripts = [
        "scripts/tests/run_compiler_test.py",
        "scripts/tests/verify_phase3.py",
        "scripts/tests/verify_performance.py",
        # 必要に応じて他の verify_*.py も追加
    ]
    
    print("\n>>> Running Verification Scripts ...")
    for script in verification_scripts:
        if os.path.exists(script):
            if not run_command(f"python {script}", f"Verification: {os.path.basename(script)}"):
                all_tests_passed = False
        else:
            print(f"⚠️ Warning: Script not found: {script}")

    if all_tests_passed:
        print("\n🎉 All tests and verifications passed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()