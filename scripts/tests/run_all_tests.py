# ファイルパス: scripts/tests/run_all_tests.py
# 日本語タイトル: マスターテストランナー (v20.5 統合版・ログ出力対応)
# 目的: プロジェクト内の全ユニットテストを一括実行し、結果をworkspace/logsに保存する。

import unittest
import sys
import os
import datetime

# プロジェクトルートの設定
# scripts/tests/run_all_tests.py から見て ../../ がルート
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# パスに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class Tee:
    """
    標準出力とファイルの両方に書き込むためのヘルパークラス
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

def setup_logging():
    """
    ログディレクトリとファイルを作成し、Teeオブジェクトを返す
    """
    # ログ保存先ディレクトリ (workspace/logs)
    log_dir = os.path.join(project_root, "workspace", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # タイムスタンプ付きログファイル名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"test_run_{timestamp}.log")
    
    print(f"📝 Logging test results to: {log_file_path}")
    
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    # 元のstdout/stderrを保持
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Teeオブジェクトを作成
    sys.stdout = Tee(original_stdout, log_file)  # type: ignore
    sys.stderr = Tee(original_stderr, log_file)  # type: ignore
    
    return log_file, original_stdout, original_stderr

def run_all_tests():
    """
    プロジェクト全体のテストを実行するランナー
    """
    # ロギングのセットアップ
    log_file, original_stdout, original_stderr = setup_logging()

    try:
        # テストディレクトリの特定 (プロジェクトルート直下の tests ディレクトリ)
        start_dir = os.path.join(project_root, "tests")
        
        if not os.path.exists(start_dir):
            print(f"❌ Error: Test directory not found at: {start_dir}")
            sys.exit(1)

        print("====================================================")
        print("   Matsushiba SNN - Master Test Suite (v17.3)")
        print(f"   Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("====================================================")
        print(f"🧪 Discovering all tests in: {start_dir}")

        # テストの探索
        # top_level_dirを指定することで、プロジェクトルートからのimportを正しく解決します
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir, pattern="test_*.py", top_level_dir=project_root)

        # テスト実行 (stream=sys.stdoutを指定してTee経由で出力)
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
        result = runner.run(suite)

        # 結果のサマリー
        print("\n" + "="*52)
        print("   Test Summary")
        print("="*52)
        print(f"Run: {result.testsRun}")
        print(f"Errors: {len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
            exit_code = 0
        else:
            print("\n❌ Some tests failed.")
            # 失敗したテストの詳細を表示（必要に応じて）
            exit_code = 1

    finally:
        # 後始末：標準出力を元に戻し、ファイルを閉じる
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    run_all_tests()