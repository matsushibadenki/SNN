# snn_research/utils/config_loader.py

import os
from typing import Optional, List
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

from snn_research.config.schema import Config


def load_config(
    config_name: str = "base_config",
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> DictConfig:
    """
    Hydraを使用して設定をロードし、Structured Config (Schema) に基づいて検証・マージを行います。

    Args:
        config_name (str): ロードする設定ファイル名（拡張子なし、または .yaml付き）
        config_path (str, optional): 設定ファイルが配置されているディレクトリの絶対パス。
                                     Noneの場合、プロジェクト標準の `configs/` を参照します。
        overrides (List[str], optional): コマンドライン引数等からのオーバーライドリスト。
                                         例: ["training.epochs=10", "model.d_model=512"]

    Returns:
        DictConfig: 検証済みの設定オブジェクト
    """

    # Hydraのグローバル状態をクリア（再入可能性のため）
    GlobalHydra.instance().clear()

    if overrides is None:
        overrides = []

    # .yaml拡張子が付いている場合は除去する（Hydraの仕様）
    if config_name.endswith('.yaml') or config_name.endswith('.yml'):
        config_name = os.path.splitext(config_name)[0]

    # デフォルトの設定ディレクトリパスを解決
    if config_path is None:
        # このファイル (snn_research/utils/config_loader.py) から見たプロジェクトルート
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, "configs")

    # config_pathが絶対パスでない場合、補正を試みるかそのままHydraに渡す
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    try:
        with initialize_config_dir(version_base=None, config_dir=config_path):
            # Hydra Compose APIを使用
            cfg = compose(config_name=config_name, overrides=overrides)

            # Structured Config (Schema) とのマージによる型チェック・補完
            # `Config` クラスの構造に従ってデフォルト値を埋める
            schema = OmegaConf.structured(Config)
            merged_cfg = OmegaConf.merge(schema, cfg)

            from typing import cast
            return cast(DictConfig, merged_cfg)

    except Exception as e:
        # エラー時のフォールバックや詳細ログが必要ならここで処理
        # 今回は上位に投げる
        raise RuntimeError(
            f"Failed to load config '{config_name}' from '{config_path}': {e}")
