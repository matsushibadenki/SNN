# ファイルパス: snn_research/cli/core_commands.py
# 日本語タイトル: SNN CLI コアコマンド定義
# 目的: テスト、クリーンアップ、ヘルスチェックなどの基本機能を提供する。

import click
import os
import sys
import shutil
import subprocess
from glob import glob
from pathlib import Path
from .utils import run_script, logger

# --- テスト機能 ---


@click.command(name="test")
@click.argument('args', nargs=-1)
def run_tests(args):
    """
    Pytestスイートを実行します。
    引数を渡すことができます（例: snn-cli test -v）。
    """
    logger.info("🧪 テストスイートを実行中 (Pytest)...")
    
    cmd = [sys.executable, "-m", "pytest", "tests/"] + list(args)
    
    try:
        # サブプロセスとしてPytestを実行
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}")
        sys.exit(1)

# --- クリーンアップ機能 ---


@click.group(name="clean")
def clean_cli():
    """プロジェクトのクリーンアップ"""
    pass


def _perform_clean(delete_models=False, delete_data=False, clean_dev_cache=True, clean_logs_flag=False, yes=False, dry_run=False):
    """
    クリーンアップ実行ロジック
    """
    if dry_run:
        logger.info("--- DRY RUN: 実際の削除は行われません ---")

    files_to_delete = []
    dirs_to_delete = []

    # 1. 開発用キャッシュ (ディレクトリごと削除)
    if clean_dev_cache:
        target_patterns = [
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
            "build", "dist", "*.egg-info", "wandb"
        ]
        root_dir = Path(".")
        for pattern in target_patterns:
            for p in root_dir.rglob(pattern):
                if p.is_dir():
                    dirs_to_delete.append(str(p))

        # 残存ファイル (.pyc 等)
        dev_file_patterns = ["*.pyc", "*.pyo",
                             "*.pyd", ".coverage", ".DS_Store"]
        for pattern in dev_file_patterns:
            for p in root_dir.rglob(pattern):
                if p.is_file():
                    files_to_delete.append(str(p))

    # 2. プロジェクト固有ディレクトリ内のファイル (中身を削除)
    project_targets = ['workspace/runs', 'workspace/results',
                       'workspace/logs', 'benchmarks/results']

    if delete_data:
        project_targets.extend(
            ['precomputed_data', 'data/prepared', 'workspace/results/temp'])

    # 削除から保護する拡張子
    protected_extensions = {'.md', '.gitignore'}
    if not delete_data:
        protected_extensions.update({'.jsonl', '.json', '.db', '.csv', '.txt'})
    if not delete_models:
        protected_extensions.update({'.pth', '.pt', '.safetensors', '.ckpt'})

    # プロジェクトディレクトリ内のスキャン
    for target_dir in project_targets:
        if os.path.isdir(target_dir):
            for filepath in glob(os.path.join(target_dir, '**', '*'), recursive=True):
                if os.path.isfile(filepath):
                    file_ext = os.path.splitext(filepath)[1]
                    if file_ext not in protected_extensions and os.path.basename(filepath) not in ['README.md', '.gitkeep']:
                        files_to_delete.append(filepath)

    # 3. ルートディレクトリのログファイル (*.log)
    if clean_logs_flag:
        for log_file in glob("*.log"):
            files_to_delete.append(log_file)

    # --- 重複排除 ---
    files_to_delete = list(set(files_to_delete))
    dirs_to_delete = list(set(dirs_to_delete))

    # --- 実行計画の表示と確認 ---
    total_items = len(dirs_to_delete) + len(files_to_delete)

    if total_items == 0:
        logger.info("削除対象の項目が見つかりません。")
        return

    logger.info("以下の項目が削除対象です:")
    for d in dirs_to_delete[:5]:
        logger.info(f"  [DIR]  {d}")
    if len(dirs_to_delete) > 5:
        logger.info(f"  ...他 {len(dirs_to_delete) - 5} ディレクトリ")

    for f in files_to_delete[:10]:
        logger.info(f"  [FILE] {f}")
    if len(files_to_delete) > 10:
        logger.info(f"  ...他 {len(files_to_delete) - 10} ファイル")

    if dry_run:
        logger.info(f"DRY RUN完了: 合計 {total_items} 項目が削除対象として検出されました。")
        return

    if not yes:
        if not click.confirm(f"合計 {total_items} 項目を削除してもよろしいですか？"):
            logger.info("キャンセルしました。")
            return

    # --- 削除実行 ---
    deleted_files_count = 0
    for f in files_to_delete:
        try:
            if os.path.exists(f):
                os.remove(f)
                deleted_files_count += 1
        except OSError as e:
            logger.warning(f"ファイルを削除できませんでした: {f} ({e})")

    deleted_dirs_count = 0
    for d in dirs_to_delete:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                deleted_dirs_count += 1
        except OSError as e:
            logger.warning(f"ディレクトリを削除できませんでした: {d} ({e})")

    empty_dirs_removed = 0
    for target_dir in project_targets:
        if os.path.isdir(target_dir):
            for root, dirs, files in os.walk(target_dir, topdown=False):
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            empty_dirs_removed += 1
                    except OSError:
                        pass

    logger.info(
        f"完了: ファイル {deleted_files_count} 個, ディレクトリ {deleted_dirs_count} 個, 空フォルダ {empty_dirs_removed} 個を削除しました。")


@clean_cli.command(name="cache")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
@click.option('--dry-run', is_flag=True, help="削除内容を表示のみ")
def clean_cache(yes, dry_run):
    """開発用キャッシュ(__pycache__等)のみ削除"""
    _perform_clean(delete_models=False, delete_data=False,
                   clean_dev_cache=True, clean_logs_flag=False, yes=yes, dry_run=dry_run)


@clean_cli.command(name="logs")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
@click.option('--dry-run', is_flag=True, help="削除内容を表示のみ")
def clean_logs(yes, dry_run):
    """ログファイル(*.log含む)とキャッシュを削除"""
    _perform_clean(delete_models=False, delete_data=False,
                   clean_dev_cache=True, clean_logs_flag=True, yes=yes, dry_run=dry_run)


@clean_cli.command(name="models")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
@click.option('--dry-run', is_flag=True, help="削除内容を表示のみ")
def clean_models(yes, dry_run):
    """ログ、キャッシュに加え、学習済みモデルも削除"""
    _perform_clean(delete_models=True, delete_data=False,
                   clean_dev_cache=True, clean_logs_flag=True, yes=yes, dry_run=dry_run)


@clean_cli.command(name="all")
@click.option('--yes', '-y', is_flag=True, help="確認なしで実行")
@click.option('--dry-run', is_flag=True, help="削除内容を表示のみ")
def clean_all(yes, dry_run):
    """すべて削除 (データ、モデル、ログ、キャッシュ含む)"""
    _perform_clean(delete_models=True, delete_data=True,
                   clean_dev_cache=True, clean_logs_flag=True, yes=yes, dry_run=dry_run)

# --- ヘルスチェック ---


@click.command(name="health-check")
def health_check():
    """健全性チェック (scripts/tests/run_project_health_check.py)"""
    script_path = "scripts/tests/run_project_health_check.py"
    run_script(script_path, [])

# --- データ管理 ---


@click.group(name="data")
def data_cli():
    """データ管理"""
    pass


@data_cli.command(name="prepare")
@click.option('--dataset', required=True)
@click.option('--output-dir', default='data/prepared')
def data_prepare(dataset, output_dir):
    script_path = "scripts/data/data_preparation.py"
    args = ["--dataset", dataset, "--output_dir", output_dir]
    run_script(script_path, args)


@data_cli.command(name="build-kb")
@click.option('--input-dir', default='data/knowledge_source')
@click.option('--output-db', default='workspace/knowledge_base.db')
def data_build_kb(input_dir, output_db):
    script_path = "scripts/data/build_knowledge_base.py"
    args = ["--input_dir", input_dir, "--output_db", output_db]
    run_script(script_path, args)


@data_cli.command(name="prep-distill")
@click.option('--task', required=True)
@click.option('--teacher-model', required=True)
@click.option('--output-dir', default='precomputed_data/distillation')
def data_prep_distill(task, teacher_model, output_dir):
    script_path = "scripts/data/prepare_distillation_data.py"
    args = ["--task", task, "--teacher_model",
            teacher_model, "--output_dir", output_dir]
    run_script(script_path, args)


def register_core_commands(cli):
    cli.add_command(run_tests)
    cli.add_command(clean_cli)
    cli.add_command(health_check)
    cli.add_command(data_cli)