# ファイルパス: app/main.py
# (動的モデルロードUI 修正 v18 - 型安全性向上)
# DIコンテナを利用した、Gradioリアルタイム対話UIの起動スクリプト

import gradio as gr  # type: ignore[import-untyped]
import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Optional, Any, Union
from omegaconf import OmegaConf, DictConfig, Container
from dependency_injector import providers
import numpy as np
from PIL import Image
import asyncio
import logging
import os 

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import AppContainer
from app.services.chat_service import ChatService
from app.services.image_classification_service import ImageClassificationService
from app.deployment import SNNInferenceEngine
from snn_research.distillation.model_registry import ModelRegistry, SimpleModelRegistry 

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- グローバル変数 ---
container = AppContainer()
available_models_dict: Dict[str, Dict[str, Any]] = {} 

# --- モデルロード関数 ---
def load_model_registry(registry_provider: providers.Provider[ModelRegistry]) -> Dict[str, Dict[str, Any]]:
    """model_registry.json を読み込み、UI用のモデル辞書を作成する"""
    print("Loading model registry...")
    try:
        registry = registry_provider()
        if not hasattr(registry, 'registry_path') or not registry.registry_path or not Path(registry.registry_path).exists():
             logger.error(f"model_registry.json not found at: {getattr(registry, 'registry_path', 'N/A')}")
             # raise FileNotFoundError # エラーで止めるのではなく空を返す
        
        models_list = asyncio.run(registry.list_models()) # 同期的に実行
        
        ui_models_dict: Dict[str, Dict[str, Any]] = {}
        for model_info in models_list:
            model_id = model_info.get("model_id")
            model_path = model_info.get("model_path") or model_info.get("path")
            config = model_info.get("config")
            
            if model_id and model_path and config:
                if isinstance(config, Container):
                    config_dict = OmegaConf.to_container(config, resolve=True)
                elif isinstance(config, dict):
                    config_dict = config
                else:
                    continue

                if not isinstance(config_dict, dict):
                     continue
                     
                task_type = "image" if "spiking_cnn" in (config_dict.get("architecture_type") or "") else "text"
                ui_models_dict[model_id] = {
                    "path": model_path,
                    "config": config_dict,
                    "task_type": task_type
                }
        print(f"✅ Found {len(ui_models_dict)} valid models in registry.")
        return ui_models_dict
    except Exception as e:
        logger.error(f"Error loading model registry: {e}")
        return {}

def load_inference_services(model_id: str) -> Tuple[Optional[ChatService], Optional[ImageClassificationService], str, Dict, Dict, Dict]:
    """選択されたモデルIDに基づいて推論サービスをロードする"""
    global available_models_dict

    if not model_id or model_id == "Select Model":
        return None, None, "Please select a model from the dropdown.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    try:
        model_info = available_models_dict.get(model_id) 
        if not model_info:
            raise KeyError(f"Model ID '{model_id}' not found in the loaded models dictionary.")

        relative_path = model_info.get("path")
        model_config_dict = model_info.get("config")
        task_type = model_info.get("task_type")
        
        model_path: Optional[str] = None
        if relative_path:
            resolved_path = Path(relative_path).resolve()
            if resolved_path.exists():
                model_path = str(resolved_path)
            else:
                logger.warning(f"Path for '{model_id}' not found on disk at '{resolved_path}'.")
        
        if not model_path or model_config_dict is None or not task_type:
            raise ValueError(f"Model info for '{model_id}' is incomplete.")

        # ベース設定とモデル設定をマージ
        config = container.config()
        full_config_dict = OmegaConf.merge(config, {"model": model_config_dict})
        OmegaConf.update(full_config_dict, "model.path", model_path, merge=True)
        
        engine_provider = container.snn_inference_engine
        
        chat_service: Optional[ChatService] = None
        image_service: Optional[ImageClassificationService] = None
        # service_instance はどちらかの型のインスタンスを持つ
        service_instance: Union[ChatService, ImageClassificationService, None] = None
        status_message = ""
        
        if task_type == "text":
            with engine_provider.override(providers.Factory(SNNInferenceEngine, config=full_config_dict)):
                service_instance = container.chat_service()
            
            # 型チェックと代入
            if isinstance(service_instance, ChatService):
                chat_service = service_instance
                status_message = f"✅ Text Model '{model_id}' loaded."
                return chat_service, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)
            else:
                 raise TypeError("Failed to initialize ChatService.")

        elif task_type == "image":
            with engine_provider.override(providers.Factory(SNNInferenceEngine, config=full_config_dict)):
                service_instance = container.image_classification_service()
            
            if isinstance(service_instance, ImageClassificationService):
                image_service = service_instance
                status_message = f"✅ Image Model '{model_id}' loaded."
                return None, image_service, status_message, gr.update(selected="image_tab"), gr.update(visible=False), gr.update(visible=True)
            else:
                 raise TypeError("Failed to initialize ImageClassificationService.")

        else:
            status_message = f"⚠️ Unknown task type '{task_type}'."
            return None, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)

    except Exception as e:
        status_message = f"❌ Error loading model '{model_id}': {e}"
        logger.error(status_message)
        return None, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)


def main():
    global available_models_dict
    
    parser = argparse.ArgumentParser(description="SNN Multi-Task Interface")
    parser.add_argument("--config", type=str, default="configs/templates/base_config.yaml")
    parser.add_argument("--chat_model_config", type=str)
    parser.add_argument("--chat_model_path", type=str)
    parser.add_argument("--cifar_model_config", type=str)
    parser.add_argument("--cifar_model_path", type=str)
    parser.add_argument("--ai_tech_model_config", type=str)
    parser.add_argument("--ai_tech_model_path", type=str)
    parser.add_argument("--summarization_model_config", type=str)
    parser.add_argument("--summarization_model_path", type=str)
    args = parser.parse_args()

    container.config.from_yaml(args.config)
    container.wire(modules=[__name__])

    available_models_dict = load_model_registry(container.model_registry)

    def add_model_from_args(model_id, config_path, model_path):
        global available_models_dict
        if config_path and model_path:
            if not Path(config_path).exists():
                return
            try:
                config_obj = OmegaConf.load(config_path)
                model_config_block = config_obj.get('model', config_obj) 
                model_config_dict = OmegaConf.to_container(model_config_block, resolve=True)
                
                if isinstance(model_config_dict, dict):
                    task_type = "image" if "spiking_cnn" in (model_config_dict.get("architecture_type") or "") else "text"
                    available_models_dict[model_id] = {
                        "path": model_path,
                        "config": model_config_dict, 
                        "task_type": task_type
                    }
            except Exception as e:
                logger.error(f"Error loading model '{model_id}': {e}")

    add_model_from_args("chat_model_default", args.chat_model_config, args.chat_model_path)
    add_model_from_args("cifar10_distilled_from_resnet18", args.cifar_model_config, args.cifar_model_path)
    add_model_from_args("最新のai技術", args.ai_tech_model_config, args.ai_tech_model_path)
    add_model_from_args("文章要約", args.summarization_model_config, args.summarization_model_path)

    model_choices = ["Select Model"] + list(available_models_dict.keys())
    initial_stats_md = "**Inference Time:** `N/A`\n**Tokens/Second:** `N/A`\n---\n**Total Spikes:** `N/A`\n**Spikes/Second:** `N/A`"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="green")) as demo:
        chat_service_state = gr.State(None)
        image_service_state = gr.State(None)

        gr.Markdown("# 🧠 SNN Multi-Task Interface")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Select Model", choices=model_choices, value=model_choices[0])
            status_textbox = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tabs() as tabs_container:
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

            with gr.TabItem("🖼️ Image Classification", id="image_tab", visible=False) as image_tab:
                with gr.Row():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_output_label = gr.Label(num_top_classes=3, label="Classification Result")
                img_classify_btn = gr.Button("Classify Image", variant="primary")

        # --- Event Handlers ---
        model_dropdown.change(
            fn=load_inference_services,
            inputs=[model_dropdown], 
            outputs=[chat_service_state, image_service_state, status_textbox, tabs_container, text_tab, image_tab],
            queue=False
        )

        def chat_clear_all(): return [], "", initial_stats_md

        def stream_chat_wrapper(message: str, history: List[List[Optional[str]]], service: Optional[ChatService]):
            if not service:
                history.append([message, "Error: Chat service is not loaded."])
                yield history, initial_stats_md
                return
            try:
                yield from service.stream_response(message, history) 
            except Exception as e:
                 logger.error(f"Error during chat stream: {e}")
                 history.append([message, f"Error: {e}"])
                 yield history, initial_stats_md

        chat_submit_event = chat_msg_textbox.submit(
            fn=stream_chat_wrapper, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], 
            outputs=[chat_chatbot, chat_stats_display], queue=False
        )
        chat_submit_event.then(fn=lambda: "", inputs=None, outputs=chat_msg_textbox)
        
        chat_button_submit_event = chat_submit_btn.click(
            fn=stream_chat_wrapper, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], 
            outputs=[chat_chatbot, chat_stats_display], queue=False
        )
        chat_button_submit_event.then(fn=lambda: "", inputs=None, outputs=chat_msg_textbox)
        
        chat_clear_btn.click(fn=chat_clear_all, inputs=None, outputs=[chat_chatbot, chat_msg_textbox, chat_stats_display], queue=False)
        
        def summarize_text(text: str, service: Optional[ChatService]):
            if not service: return "Error: Service not loaded.", initial_stats_md
            full_response = ""
            stats_md = initial_stats_md
            try:
                for hist, stats in service.stream_response(text, []):
                    if hist and hist[-1][1]:
                        full_response = hist[-1][1]
                    stats_md = stats
            except Exception as e:
                full_response = f"Error: {e}"
            return full_response, stats_md

        sum_summarize_btn.click(
            fn=summarize_text, inputs=[sum_input_textbox, chat_service_state], 
            outputs=[sum_output_textbox, sum_stats_display], queue=False
        )

        def classify_image(image: Any, service: Optional[ImageClassificationService]):
             if not service: return {"Error": 1.0, "Service not loaded": 0.0}
             try: return service.predict(image)
             except Exception as e: return {"Error": 1.0, str(e): 0.0}

        img_classify_btn.click(
            fn=classify_image, inputs=[img_input, image_service_state], 
            outputs=[img_output_label], queue=False
        )

    config_obj = container.config()
    server_port = config_obj.get('app', {}).get('server_port', 7860)
    server_name = config_obj.get('app', {}).get('server_name', '127.0.0.1')
    
    demo.launch(server_name=server_name, server_port=server_port)

if __name__ == "__main__":
    main()
