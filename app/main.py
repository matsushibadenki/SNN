# ファイルパス: app/main.py
# タイトル: DIコンテナ・Gradio UI起動スクリプト (Clean UI版)

import gradio as gr
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class SNNInterfaceApp:
    """Gradio UIの状態とDIコンテナのオーケストレーター。"""
    
    def _add_model_entry(self, model_id: str, path: str, config: Any):
        """モデル情報の解析と登録。"""
        if not all([model_id, path, config]): return

        config_dict = OmegaConf.to_container(config, resolve=True) if hasattr(config, '__dict__') else config
        if not isinstance(config_dict, dict): return

        # アーキテクチャ名からタスクを推論
        arch = str(config_dict.get("architecture_type", "")).lower()
        task_type = "image" if any(x in arch for x in ["cnn", "visual", "vision"]) else "text"

        self.available_models_dict[model_id] = {
            "path": path, "config": config_dict, "task_type": task_type
        }

    def create_ui(self) -> gr.Blocks:
        """UIコンポーネントの配置とバインディング。"""
        with gr.Blocks(title="SNN Multi-Task Interface", theme=gr.themes.Soft()) as demo:
            chat_service_state = gr.State(None)
            image_service_state = gr.State(None)
            
            gr.Markdown("# 🧠 SNN Intelligence Hub")
            
            with gr.Row():
                model_drop = gr.Dropdown(
                    label="Active Model", 
                    choices=["Select Model"] + list(self.available_models_dict.keys()),
                    value="Select Model"
                )
                status_out = gr.Textbox(label="System Status", interactive=False)

            # ... (タブ定義は既存のロジックを継承) ...

            # イベントバインディングの整理
            model_drop.change(
                fn=self.load_inference_services,
                inputs=[model_drop],
                outputs=[chat_service_state, image_service_state, status_out]
            )
            
        return demo
