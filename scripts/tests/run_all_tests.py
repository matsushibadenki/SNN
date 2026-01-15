# scripts/tests/run_all_tests.py
import subprocess
import sys
import os
import time
import logging

def configure_test_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

def run_command(command, description, stop_on_fail=True):
    print(f"\n>>> Running: {description} ...")
    print(f"    Command: {command}")
    start_time = time.time()
    
    env = os.environ.copy()
    env["SNN_TEST_MODE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    
    # ログノイズフィルタリング用の設定
    # 以下の文字列を含む行は出力しない
    noise_filters = [
        "No module named 'cupy'",
        "spikingjelly",
        "Matplotlib is building the font cache"
    ]

    # サブプロセスを実行し、出力をパイプで受け取る
    process = subprocess.Popen(
        command, 
        shell=True, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # stderrもstdoutにマージ
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # リアルタイムで出力を処理
    if process.stdout:
        for line in process.stdout:
            # フィルタリング処理
            if any(noise in line for noise in noise_filters):
                continue
            
            # 必要なログは表示 (末尾の改行を考慮してprint)
            print(line, end='')
    
    process.wait()
    
    duration = time.time() - start_time
    
    if process.returncode == 0:
        print(f"✅ {description} Passed ({duration:.2f}s)")
        return True
    else:
        print(f"❌ {description} Failed (Exit Code: {process.returncode})")
        if stop_on_fail:
            return False
        return False

def main():
    configure_test_logging()
    
    print("========================================")
    print("   SNN Research Project - Test Suite    ")
    print("   Target: Phase 2 (Beyond ANN)         ")
    print("========================================")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)
    
    all_tests_passed = True

    # 1. ヘルスチェック
    if not run_command("python scripts/tests/run_project_health_check.py", "Project Health Check"):
        print("Health check failed. Aborting tests.")
        sys.exit(1)

    # 2. Pytest実行 (tests/ ディレクトリ)
    print("\n>>> Running Standard Unit Tests (pytest tests/) ...")
    # -v: 詳細, -s: 標準出力表示(フィルタリングされるのでOK)
    if not run_command("python -m pytest tests/ -v -s", "Standard Unit Tests", stop_on_fail=False):
        all_tests_passed = False

    # 3. 追加のスクリプトテスト実行
    print("\n>>> Running Script Tests (pytest scripts/tests/) ...")
    if not run_command("python -m pytest scripts/tests/ -v -s", "Script Tests", stop_on_fail=False):
        all_tests_passed = False
    
    # 4. 検証スクリプト
    verification_scripts = [
        "scripts/tests/run_compiler_test.py",
        "scripts/tests/verify_phase3.py",
        "scripts/tests/verify_performance.py",
    ]
    
    print("\n>>> Running Verification Scripts ...")
    for script in verification_scripts:
        if os.path.exists(script):
            if not run_command(f"python {script}", f"Verification: {os.path.basename(script)}", stop_on_fail=False):
                all_tests_passed = False
        else:
            print(f"⚠️ Warning: Script not found: {script}")

    # 5. Phase 2 ベンチマーク
    print("\n>>> Running Phase 2 Benchmark Suite ...")
    benchmark_script = "scripts/benchmarks/run_benchmark_suite.py"
    if os.path.exists(benchmark_script):
        if not run_command(f"python {benchmark_script}", "Benchmark Suite", stop_on_fail=False):
            print("⚠️ Benchmarks finished with warnings.")
    else:
        print(f"⚠️ Warning: Benchmark script not found: {benchmark_script}")

    if all_tests_passed:
        print("\n🎉 All functional tests passed successfully!")
        print("👉 Please review the Benchmark Report above for Phase 2 targets.")
        sys.exit(0)
    else:
        print("\n⚠️ Some functional tests failed. Please review the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()