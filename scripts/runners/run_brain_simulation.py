# ファイルパス: scripts/runners/run_brain_simulation.py

import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_brain_simulation.py
# (修正)
# 修正: DIコンテナがモデルアーキテクチャ設定（small.yaml）を読み込むように修正し、
#       実行時エラーを解消。
# 改善(v2): コマンドラインから単一の入力を受け取れるようにargparseを導入。
#
# 修正 (v3):
# - 健全性チェック (health-check) での `終了コード: 2` エラーを解消。
# - DIコンテナ (dependency-injector) が `dict` を期待するのに対し、
#   `container.config.from_yaml` が `OmegaConf` オブジェクトを
#   誤ってロードしようとしていた（あるいはその逆）問題を修正。
# - `OmegaConf.load` で設定を読み込み、`OmegaConf.to_container` で
#   標準の `dict` に変換してから `container.config.from_dict` で設定する
#   堅牢な方法に変更。

import sys
from pathlib import Path
import time
import argparse
# --- ▼ 修正 (v3): OmegaConf をインポート ▼ ---
from omegaconf import OmegaConf, DictConfig
# --- ▲ 修正 (v3) ▲ ---

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import BrainContainer

def main():
    """
    DIコンテナを使って人工脳を初期化し、シミュレーションを実行する。
    """
    parser = argparse.ArgumentParser(description="Artificial Brain Simulation Runner")
    parser.add_argument("--prompt", type=str, help="人工脳への単一の入力テキスト。指定しない場合はデモを実行します。")
    # --- ▼ 修正 (v3): model_config 引数を argparse に追加 ▼ ---
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/small.yaml",
        help="モデルアーキテクチャ設定ファイルのパス。"
    )
    # --- ▲ 修正 (v3) ▲ ---
    args = parser.parse_args()

    # --- ▼ 修正 (v3): 設定ファイルのロード方法を堅牢化 ▼ ---
    # 1. DIコンテナを初期化
    container = BrainContainer()

    # 2. OmegaConfで設定を読み込み、マージ
    try:
        base_cfg = OmegaConf.load("configs/templates/base_config.yaml")
        model_cfg = OmegaConf.load(args.model_config)
        merged_cfg = OmegaConf.merge(base_cfg, model_cfg)
        
        # 3. DIコンテナには標準の dict として設定を渡す
        #    (OmegaConf.to_container で dict に変換)
        config_dict = OmegaConf.to_container(merged_cfg, resolve=True)
        if isinstance(config_dict, dict):
            container.config.from_dict(config_dict)
        else:
            raise TypeError("Loaded config is not a dictionary.")
            
    except FileNotFoundError as e:
        print(f"❌ エラー: 設定ファイルが見つかりません: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラー: 設定ファイルの読み込みに失敗しました: {e}")
        sys.exit(1)
    # --- ▲ 修正 (v3) ▲ ---

    # 4. コンテナから完成品の人工脳インスタンスを取得
    try:
        brain = container.artificial_brain()
    except Exception as e:
        print(f"❌ エラー: 人工脳の構築に失敗しました。DIコンテナの設定を確認してください。")
        print(f"詳細: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2) # 終了コード 2

    # 5. シミュレーションの実行
    if args.prompt:
        # 単一の入力で実行
        print(f"--- Running single cognitive cycle for input: '{args.prompt}' ---")
        brain.run_cognitive_cycle(args.prompt)
    else:
        # デモモード
        print("--- Running demonstration with multiple inputs ---")
        inputs = [
            "素晴らしい発見だ！これは成功に繋がるだろう。",
            "エラーが発生しました。システムに問題があるようです。",
            "今日は穏やかな一日だ。"
        ]
        for text_input in inputs:
            brain.run_cognitive_cycle(text_input)
            time.sleep(1) # 各サイクルの間に少し待機

if __name__ == "__main__":
    main()