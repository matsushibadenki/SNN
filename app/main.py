# ファイルパス: app/main.py
# タイトル: DIコンテナ・Gradio UI起動スクリプト (Refactored for Robustness)
# 目的: DIコンテナを利用してGradioによるリアルタイム対話UIを起動する。
# 修正: グローバル変数の排除、クラスベースの状態管理、エラーハンドリングの強化。

import gradio as gr  # type: ignore[import-untyped]
import argparse
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union
from omegaconf import OmegaConf, Container
from dependency_injector import providers
import asyncio

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.containers import AppContainer
from app.services.chat_service import ChatService
from app.services.image_classification_service import ImageClassificationService
from app.deployment import SNNInferenceEngine
from snn_research.distillation.model_registry import ModelRegistry

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SNNInterfaceApp:
    """
    SNN Interface Application Class
    Gradio UIの状態とDIコンテナを管理する。
    """
    def __init__(self, config_path: str, cli_args: argparse.Namespace):
        self.container = AppContainer()
        # 設定ファイルのロード
        self.container.config.from_yaml(config_path)
        self.container.wire(modules=[__name__])
        
        self.available_models_dict: Dict[str, Dict[str, Any]] = {}
        self.cli_args = cli_args
        
        # 初期化
        self._initialize_models()

    def _initialize_models(self):
        """モデルレジストリとCLI引数からモデル一覧を初期化する"""
        logger.info("Initializing models...")
        
        # 1. モデルレジストリからのロード
        try:
            registry = self.container.model_registry()
            if hasattr(registry, 'registry_path') and registry.registry_path and Path(registry.registry_path).exists():
                models_list = asyncio.run(registry.list_models())
                
                for model_info in models_list:
                    self._add_model_entry(
                        model_id=model_info.get("model_id"),
                        path=model_info.get("model_path") or model_info.get("path"),
                        config=model_info.get("config")
                    )
            else:
                logger.warning(f"Model registry path not found or invalid.")
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")

        # 2. CLI引数からのロード
        self._add_model_from_args("chat_model_default", self.cli_args.chat_model_config, self.cli_args.chat_model_path)
        self._add_model_from_args("cifar10_distilled", self.cli_args.cifar_model_config, self.cli_args.cifar_model_path)
        self._add_model_from_args("ai_tech_model", self.cli_args.ai_tech_model_config, self.cli_args.ai_tech_model_path)
        self._add_model_from_args("summarization", self.cli_args.summarization_model_config, self.cli_args.summarization_model_path)
        self._add_model_from_args("CLI_Selected_Model", self.cli_args.model_config, self.cli_args.model_path)

        logger.info(f"✅ Loaded {len(self.available_models_dict)} models available for interface.")

    def _add_model_from_args(self, model_id: str, config_path: Optional[str], model_path: Optional[str]):
        """CLI引数からモデルを追加するヘルパー"""
        if config_path and model_path:
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                logger.warning(f"Config file not found: {config_path}")
                return
            try:
                config_obj = OmegaConf.load(config_path_obj)
                # 'model'キーがあればその中身、なければ全体を使用
                model_config_block = config_obj.get('model', config_obj)
                self._add_model_entry(model_id, model_path, model_config_block)
            except Exception as e:
                logger.error(f"Error adding CLI model '{model_id}': {e}")

    def _add_model_entry(self, model_id: Optional[str], path: Optional[str], config: Any):
        """モデル辞書への登録処理"""
        if not model_id or not path or not config:
            return

        if isinstance(config, Container):
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            config_dict = config
        else:
            return

        if not isinstance(config_dict, dict):
            return

        # タスクタイプの推定
        arch_type = config_dict.get("architecture_type", "")
        if "cnn" in str(arch_type).lower() or "visual" in str(arch_type).lower():
            task_type = "image"
        else:
            task_type = "text"

        self.available_models_dict[model_id] = {
            "path": path,
            "config": config_dict,
            "task_type": task_type
        }

    def load_inference_services(self, model_id: str) -> Tuple[Optional[ChatService], Optional[ImageClassificationService], str, Dict, Dict, Dict]:
        """UIコールバック: 選択されたモデルをロードしてサービスをインスタンス化する"""
        if not model_id or model_id == "Select Model":
            return None, None, "Please select a model.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        try:
            model_info = self.available_models_dict.get(model_id)
            if not model_info:
                raise KeyError(f"Model ID '{model_id}' not found.")

            relative_path = model_info["path"]
            model_config_dict = model_info["config"]
            task_type = model_info["task_type"]

            # パスの絶対パス化
            resolved_path = Path(relative_path).resolve()
            if not resolved_path.exists():
                 logger.warning(f"Model file not found at: {resolved_path}")
                 # 続行を試みる（パスが間違っているだけの可能性もあるため）

            model_path = str(resolved_path)

            # 設定のマージ
            base_config = self.container.config()
            full_config_dict = OmegaConf.merge(base_config, {"model": model_config_dict})
            OmegaConf.update(full_config_dict, "model.path", model_path, merge=True)

            # エンジンのオーバーライドとサービス生成
            engine_provider = self.container.snn_inference_engine
            
            chat_service: Optional[ChatService] = None
            image_service: Optional[ImageClassificationService] = None
            status_message = ""

            # DIコンテナのプロバイダを一時的にオーバーライド
            with engine_provider.override(providers.Factory(SNNInferenceEngine, config=full_config_dict)):
                
                if task_type == "text":
                    service = self.container.chat_service()
                    if isinstance(service, ChatService):
                        chat_service = service
                        status_message = f"✅ Text Model '{model_id}' loaded."
                        return chat_service, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)
                    else:
                        raise TypeError("Failed to create ChatService")

                elif task_type == "image":
                    service = self.container.image_classification_service()
                    if isinstance(service, ImageClassificationService):
                        image_service = service
                        status_message = f"✅ Image Model '{model_id}' loaded."
                        return None, image_service, status_message, gr.update(selected="image_tab"), gr.update(visible=False), gr.update(visible=True)
                    else:
                        raise TypeError("Failed to create ImageClassificationService")
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            msg = f"❌ Error loading model '{model_id}': {str(e)}"
            logger.error(msg)
            return None, None, msg, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)

    def create_ui(self) -> gr.Blocks:
        """Gradio UIの構築"""
        model_choices = ["Select Model"] + list(self.available_models_dict.keys())
        initial_stats_md = "**Inference Time:** `N/A`\n**Tokens/Second:** `N/A`\n---\n**Total Spikes:** `N/A`\n**Spikes/Second:** `N/A`"

        with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="green")) as demo:
            # 状態変数
            chat_service_state = gr.State(None)
            image_service_state = gr.State(None)

            gr.Markdown("# 🧠 SNN Multi-Task Interface")

            with gr.Row():
                model_dropdown = gr.Dropdown(label="Select Model", choices=model_choices, value=model_choices[0])
                status_textbox = gr.Textbox(label="Status", interactive=False)

            with gr.Tabs() as tabs_container:
                # --- Text Tab ---
                with gr.TabItem("💬 Text / Chat", id="text_tab") as text_tab:
                    with gr.Row():
                        chat_chatbot = gr.Chatbot(label="SNN Chat", height=500)
                        chat_stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")
                    with gr.Row():
                        chat_msg_textbox = gr.Textbox(show_label=False, placeholder="メッセージを入力...", container=False, scale=6)
                        chat_submit_btn = gr.Button("Send", variant="primary", scale=1)
                        chat_clear_btn = gr.Button("Clear", scale=1)
                    
                    with gr.Accordion("Summarization", open=False):
                        with gr.Row():
                            sum_input_textbox = gr.Textbox(label="Input Text", lines=10)
                            sum_output_textbox = gr.Textbox(label="Summary", lines=10, interactive=False)
                        sum_summarize_btn = gr.Button("Summarize", variant="primary")
                        sum_stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

                # --- Image Tab ---
                with gr.TabItem("🖼️ Image Classification", id="image_tab", visible=False) as image_tab:
                    with gr.Row():
                        img_input = gr.Image(type="pil", label="Upload Image")
                        img_output_label = gr.Label(num_top_classes=3, label="Classification Result")
                    img_classify_btn = gr.Button("Classify Image", variant="primary")

            # --- Callbacks ---
            
            # モデル切り替え
            model_dropdown.change(
                fn=self.load_inference_services,
                inputs=[model_dropdown],
                outputs=[chat_service_state, image_service_state, status_textbox, tabs_container, text_tab, image_tab],
                queue=False
            )

            # Chat Logic
            def stream_chat(message: str, history: List, service: Optional[ChatService]):
                if not service:
                    history.append([message, "⚠️ Error: Model is not loaded. Please select a text model."])
                    yield history, initial_stats_md
                    return
                try:
                    yield from service.stream_response(message, history)
                except Exception as e:
                    logger.error(f"Chat Error: {e}")
                    history.append([message, f"❌ Internal Error: {str(e)}"])
                    yield history, initial_stats_md

            def clear_chat():
                return [], "", initial_stats_md

            chat_submit_btn.click(
                fn=stream_chat, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], 
                outputs=[chat_chatbot, chat_stats_display]
            ).then(lambda: "", outputs=chat_msg_textbox)

            chat_msg_textbox.submit(
                fn=stream_chat, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], 
                outputs=[chat_chatbot, chat_stats_display]
            ).then(lambda: "", outputs=chat_msg_textbox)
            
            chat_clear_btn.click(fn=clear_chat, outputs=[chat_chatbot, chat_msg_textbox, chat_stats_display])

            # Summarization Logic
            def run_summary(text: str, service: Optional[ChatService]):
                if not service:
                    return "⚠️ Error: Model not loaded.", initial_stats_md
                full_res = ""
                stats = initial_stats_md
                try:
                    # ストリームの結果を最後まで回して取得
                    for hist, st in service.stream_response(text, []):
                        if hist: full_res = hist[-1][1]
                        stats = st
                except Exception as e:
                    full_res = f"❌ Error: {str(e)}"
                return full_res, stats

            sum_summarize_btn.click(
                fn=run_summary, inputs=[sum_input_textbox, chat_service_state],
                outputs=[sum_output_textbox, sum_stats_display]
            )

            # Image Logic
            def run_classify(img, service: Optional[ImageClassificationService]):
                if not service:
                    return {"⚠️ Error: Model not loaded": 0.0}
                if img is None:
                    return {"⚠️ Error: No image uploaded": 0.0}
                try:
                    return service.predict(img)
                except Exception as e:
                    logger.error(f"Image Error: {e}")
                    return {f"❌ Error: {str(e)}": 0.0}

            img_classify_btn.click(
                fn=run_classify, inputs=[img_input, image_service_state],
                outputs=[img_output_label]
            )

        return demo

def main():
    parser = argparse.ArgumentParser(description="SNN Multi-Task Interface")
    parser.add_argument("--config", type=str, default="configs/templates/base_config.yaml")
    parser.add_argument("--server_name", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)
    
    # Model arguments
    parser.add_argument("--chat_model_config", type=str)
    parser.add_argument("--chat_model_path", type=str)
    parser.add_argument("--cifar_model_config", type=str)
    parser.add_argument("--cifar_model_path", type=str)
    parser.add_argument("--ai_tech_model_config", type=str)
    parser.add_argument("--ai_tech_model_path", type=str)
    parser.add_argument("--summarization_model_config", type=str)
    parser.add_argument("--summarization_model_path", type=str)
    parser.add_argument("--model_config", type=str, help="CLI specified model config")
    parser.add_argument("--model_path", type=str, help="CLI specified model path")
    
    args = parser.parse_args()

    # アプリケーション初期化
    app = SNNInterfaceApp(args.config, args)
    
    # サーバー設定の決定 (CLI優先、次にConfig)
    config_obj = app.container.config()
    server_name = args.server_name or config_obj.get('app', {}).get('server_name', '127.0.0.1')
    server_port = args.server_port or config_obj.get('app', {}).get('server_port', 7860)

    # UI起動
    ui = app.create_ui()
    ui.launch(server_name=server_name, server_port=server_port)

if __name__ == "__main__":
    main()
