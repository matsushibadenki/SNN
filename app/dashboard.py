# ファイルパス: app/dashboard.py
# Title: Brain v20 Dashboard (Enhanced & Type Fix)
# Description: fMRI風可視化機能を追加し、pandasのmypyエラーを修正したダッシュボード。

import gradio as gr # type: ignore[import-untyped]
import sys
from pathlib import Path
import json
import pandas as pd # type: ignore[import-untyped] # 修正: スタブ欠落エラーを抑制

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

print("🧠 Initializing Artificial Brain for Dashboard...")
container = BrainContainer()
container.config.from_yaml("configs/templates/base_config.yaml")
container.config.from_yaml("configs/models/small.yaml")
brain = container.artificial_brain()

# --- CSS / HTML Generator for Brain Map ---
def generate_brain_map_html(active_modules: list, valence: float, arousal: float) -> str:
    """脳のモジュール活性状態を可視化"""
    def get_opacity(module_name):
        return "1.0" if module_name in active_modules else "0.3"
    
    def get_glow(module_name):
        return "box-shadow: 0 0 15px #00ff00;" if module_name in active_modules else ""

    bg_color = f"rgba({int(arousal * 50)}, {int(valence * 50)}, 100, 0.1)"
    
    html = f"""
    <style>
        .brain-container {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            padding: 20px;
            background-color: {bg_color};
            border-radius: 10px;
            height: 250px;
        }}
        .module-box {{
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        .cortex {{ background-color: #3f51b5; }}
        .system1 {{ background-color: #009688; }}
        .system2 {{ background-color: #ff9800; }}
        .amygdala {{ background-color: #e91e63; }}
        .hippocampus {{ background-color: #9c27b0; }}
    </style>
    
    <div class="brain-container">
        <div class="module-box cortex" style="opacity: {get_opacity('visual_cortex')}; {get_glow('visual_cortex')}">Visual Cortex</div>
        <div class="module-box system1" style="opacity: {get_opacity('system1')}; {get_glow('system1')}">System 1 (Intuition)</div>
        <div class="module-box amygdala" style="opacity: {get_opacity('amygdala')}; {get_glow('amygdala')}">Amygdala</div>
        <div class="module-box system2" style="opacity: {get_opacity('reasoning_engine')}; {get_glow('reasoning_engine')}">System 2 (Reasoning)</div>
        <div class="module-box hippocampus" style="opacity: {get_opacity('hippocampus')}; {get_glow('hippocampus')}">Hippocampus</div>
    </div>
    """
    return html

def process_brain_cycle(user_input: str):
    """脳サイクル実行とダッシュボード更新"""
    if not user_input:
        return pd.DataFrame(), "Waiting...", "", "", "", "<div>Inactive</div>"

    # Brain実行
    brain.run_cognitive_cycle(user_input)
    
    # 状態取得
    active_modules = ["visual_cortex", "system1"]
    
    # [mypy修正] upload_to_workspace で入った情報を正しく取得
    amygdala_info = brain.workspace.get_information("amygdala")
    valence = 0.5
    arousal = 0.5
    if isinstance(amygdala_info, dict):
        valence = amygdala_info.get('valence', 0.5)
        arousal = amygdala_info.get('arousal', 0.5)
    
    # 行動の取得
    action_str = str(brain.basal_ganglia.selected_action)
    if "thinking" in action_str:
        active_modules.append("reasoning_engine")
    
    # コンテキスト情報の取得
    conscious_content = brain.workspace.conscious_broadcast_content
    
    # ログデータ
    log_data = [
        {"Time": "T", "Event": "STIMULUS", "Payload": user_input},
        {"Time": "T+1", "Event": "BROADCAST", "Payload": str(conscious_content)},
        {"Time": "T+2", "Event": "DECISION", "Payload": action_str}
    ]
    df_log = pd.DataFrame(log_data)

    # HTML生成
    brain_map = generate_brain_map_html(active_modules, valence, arousal)
    
    return (
        df_log,
        action_str,
        json.dumps(conscious_content, indent=2, ensure_ascii=False),
        "Memory active", # WM
        "Nodes synchronized", # Cortex info
        brain_map
    )
    
# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="Brain v20 Dashboard") as demo:
    gr.Markdown("# 🧠 Brain v20: Live Thought Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Sensory Input")
            run_btn = gr.Button("⚡ Inject Stimulus", variant="primary")
            action_display = gr.Textbox(label="Action")
            conscious_display = gr.Code(label="Conscious Stream", language="json")

        with gr.Column(scale=2):
            brain_map_display = gr.HTML(label="Neural Activity Map")
            log_table = gr.Dataframe(headers=["Time", "Event", "Payload"])
            cortex_display = gr.Textbox(label="Long-Term Memory")

    run_btn.click(
        fn=process_brain_cycle,
        inputs=[input_text],
        outputs=[log_table, action_display, conscious_display, gr.Textbox(), cortex_display, brain_map_display],
        queue=False 
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)