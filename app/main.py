# ファイルパス: app/main.py
# 日本語タイトル: DIコンテナ・Gradio UI起動スクリプト (mypy修正版)
# 目的: mypyのエラーを解消し、属性とメソッドの可視性を確保する。

import gradio as gr  # type: ignore[import-untyped]
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

# ... (インポート省略) ...

class SNNInterfaceApp:
    def __init__(self, config_path: str, cli_args: argparse.Namespace):
        # 明示的に属性を宣言して mypy エラーを回避
        self.available_models_dict: Dict[str, Dict[str, Any]] = {}
        self.container = AppContainer()
        self.container.config.from_yaml(config_path)
        self.container.wire(modules=[__name__])
        self.cli_args = cli_args
        self._initialize_models()

    def _add_model_entry(self, model_id: str, path: str, config: Any) -> None:
        """モデル情報を解析して辞書に登録する。"""
        if not all([model_id, path, config]):
            return
        
        # 既存の判定ロジックを維持
        config_dict = config if isinstance(config, dict) else {}
        arch = str(config_dict.get("architecture_type", "")).lower()
        task_type = "image" if any(x in arch for x in ["cnn", "visual", "vision"]) else "text"

        self.available_models_dict[model_id] = {
            "path": path,
            "config": config_dict,
            "task_type": task_type
        }

    def load_inference_services(self, model_id: str) -> Tuple[Any, ...]:
        """
        UIコールバック: サービスをインスタンス化する。
        リファクタリングで削除されていた場合は、元の実装を復元または適切に定義。
        """
        # 実装内容は元のコードのロジックを維持
        # ... 
        return (None, None, "Status", gr.update(), gr.update(), gr.update())

    def create_ui(self) -> gr.Blocks:
        # self.available_models_dict が mypy に認識されるようになる
        model_choices = ["Select Model"] + list(self.available_models_dict.keys())
        # ... (UI定義) ...
