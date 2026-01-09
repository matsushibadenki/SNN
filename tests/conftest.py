# ファイルパス: tests/conftest.py
# 日本語タイトル: Pytest用設定ファイル
# 目的: テスト実行時にプロジェクトルートへのパスを通し、モジュール解決を確実にする。

import sys
import os
import pytest

# プロジェクトルートの絶対パスを取得 (tests/../)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sys.pathの先頭に追加して、インストールされていない状態でもインポート可能にする
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture(scope="session")
def project_root():
    """プロジェクトルートパスを提供するフィクスチャ"""
    return PROJECT_ROOT