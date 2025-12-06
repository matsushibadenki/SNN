# ファイルパス: app/dashboard.py
# (修正: mypy attr-defined エラー修正)
# Title: Artificial Brain Dashboard
# Description: 人工脳の内部状態をリアルタイム可視化するGUI。
# 修正: Gradioのキューエラー回避のため、イベントリスナーに queue=False を追加。
# 修正: ArtificialBrainの初期化にmotivation_systemを渡すように変更。

import gradio as gr # type: ignore[import-untyped]
import sys
from pathlib import Path
import json

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

print("🧠 Initializing Artificial Brain for Dashboard...")
container = BrainContainer()
container.config.from_yaml("configs/templates/base_config.yaml")
container.config.from_yaml("configs/models/small.yaml")
brain = container.artificial_brain()

def process_brain_cycle(user_input: str):
    if not user_input:
        return "Please enter text.", {}, "", "", "", ""

    brain.run_cognitive_cycle(user_input)

    # 1. 感情状態
    amygdala_info = brain.workspace.get_information("amygdala")
    # 修正: motivation_system を使用
    internal_state = brain.motivation_system.get_internal_state()
    
    emotion_summary = {
        "Valence": amygdala_info.get('valence', 0.0) if amygdala_info else 0.0,
        "Arousal": amygdala_info.get('arousal', 0.0) if amygdala_info else 0.0,
        "Curiosity": f"{internal_state.get('curiosity', 0.0):.2f}",
        "Boredom": f"{internal_state.get('boredom', 0.0):.2f}"
    }

    # 2. 意識の内容
    conscious_content = brain.workspace.conscious_broadcast_content
    conscious_str = json.dumps(conscious_content, indent=2, ensure_ascii=False) if conscious_content else "(No dominant thought)"

    # 3. 意思決定
    action = brain.basal_ganglia.selected_action
    action_str = action.get('action', 'No Action') if action else "No Action"
    
    # 4. 記憶
    wm_episodes = list(brain.hippocampus.working_memory)
    wm_str = "\n".join([f"- {str(ep.get('source_input', ''))[:30]}..." for ep in wm_episodes[-5:]])
    
    # --- ▼ 修正: get_all_knowledge() を使用 ▼ ---
    kg_size = len(brain.cortex.get_all_knowledge())
    # --- ▲ 修正 ▲ ---
    
    system_log = f"Cycle: {brain.cycle_count}\nAction Selected: {action_str}\nKnowledge Graph Nodes: {kg_size}"

    return (system_log, emotion_summary, conscious_str, action_str, wm_str, f"Nodes: {kg_size}")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🧠 Artificial Brain Dashboard")
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Input", placeholder="Talk to the brain...")
            run_btn = gr.Button("Run Cycle", variant="primary")
            log_output = gr.Textbox(label="Log", interactive=False, lines=5)
        with gr.Column(scale=2):
            with gr.Row():
                emotion_display = gr.JSON(label="Emotion & Motivation")
                with gr.Column():
                    conscious_display = gr.Code(label="Consciousness", language="json")
                    action_display = gr.Textbox(label="Action")
            with gr.Row():
                wm_display = gr.Textbox(label="Working Memory", lines=5)
                cortex_display = gr.Textbox(label="Long-Term Knowledge", lines=5)

    run_btn.click(
        fn=process_brain_cycle,
        inputs=[input_text],
        outputs=[
            log_output,
            emotion_display,
            conscious_display,
            action_display,
            wm_display,
            cortex_display
        ],
        queue=False 
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
