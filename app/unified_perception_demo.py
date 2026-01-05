# ファイルパス: app/unified_perception_demo.py
# 日本語タイトル: Unified Perception Interactive Demo (Gradio) [Type Fixed]
# 目的・内容:
#   mypyエラー修正版。
#   - gradioインポート時の型チェックを無視設定に追加。
#   - trainメソッド内の変数 log に型ヒントを追加。

import gradio as gr  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from typing import Tuple, Dict  # Added: List

# プロジェクトルートをパスに追加
# プロジェクトルートへのパス追加コードを削除しました。

from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex
from snn_research.learning_rules.stdp import STDP
from snn_research.io.universal_encoder import UniversalSpikeEncoder


class BrainSystem:
    """デモ用の脳システムラッパー"""

    def __init__(self):
        self.device = "cpu"
        self.time_steps = 50

        # Dimensions
        self.dim_visual = 100
        self.dim_audio = 50
        self.reservoir_size = 500

        # Hyperparameters (Optimized from Phase 9-10)
        self.input_scale = 20.0
        self.threshold = 0.5

        # Initialize Components
        self.stdp_rule = STDP(learning_rate=0.02, a_plus=0.1,
                              a_minus=0.05, tau_trace=20.0)

        self.lac = LiquidAssociationCortex(
            num_visual_inputs=self.dim_visual,
            num_audio_inputs=self.dim_audio,
            num_text_inputs=10,
            num_somato_inputs=10,
            reservoir_size=self.reservoir_size,
            sparsity=0.2,
            tau=5.0,
            threshold=self.threshold,
            input_scale=self.input_scale,
            learning_rule=self.stdp_rule
        ).to(self.device)

        self.encoder = UniversalSpikeEncoder(
            time_steps=self.time_steps, device=self.device)

        # Concept Prototypes (Fixed Patterns)
        torch.manual_seed(42)
        self.vis_A = (torch.rand(1, self.dim_visual) > 0.7).float()
        self.aud_A = (torch.rand(1, self.dim_audio) > 0.7).float()
        self.vis_B = (torch.rand(1, self.dim_visual) > 0.7).float()
        self.aud_B = (torch.rand(1, self.dim_audio) > 0.7).float()

        # Training State
        self.is_trained = False
        self.training_log = "Not trained yet."

        # Target Activities (for similarity check)
        self.target_A_activity = None
        self.target_B_activity = None

        # 初期状態のターゲット活動を記録
        self._record_targets()

    def _record_targets(self):
        """概念A, Bに対応する理想的なリザーバ活動を記録"""
        self.lac.eval()

        # Record A (Vis A + Aud A)
        self.lac.reset_state()
        spikes_A = []
        vis_spikes = self.encoder.encode(self.vis_A, 'image', 'rate')
        aud_spikes = self.encoder.encode(self.aud_A, 'audio', 'rate')
        for t in range(self.time_steps):
            out = self.lac(
                visual_spikes=vis_spikes[:, t], audio_spikes=aud_spikes[:, t])
            spikes_A.append(out.detach())
        self.target_A_activity = torch.stack(
            spikes_A).squeeze(1).mean(dim=0)  # (Neurons,)

        # Record B (Vis B + Aud B)
        self.lac.reset_state()
        spikes_B = []
        vis_spikes = self.encoder.encode(self.vis_B, 'image', 'rate')
        aud_spikes = self.encoder.encode(self.aud_B, 'audio', 'rate')
        for t in range(self.time_steps):
            out = self.lac(
                visual_spikes=vis_spikes[:, t], audio_spikes=aud_spikes[:, t])
            spikes_B.append(out.detach())
        self.target_B_activity = torch.stack(spikes_B).squeeze(1).mean(dim=0)

    def train(self, epochs: int) -> str:
        """連合学習を実行"""
        self.lac.train()
        # 修正: 型ヒントを追加
        # log: List[str] = []

        for epoch in range(epochs):
            # Train Concept A
            self.lac.reset_state()
            vis_spikes = self.encoder.encode(self.vis_A, 'image', 'rate')
            aud_spikes = self.encoder.encode(self.aud_A, 'audio', 'rate')
            for t in range(self.time_steps):
                self.lac(
                    visual_spikes=vis_spikes[:, t], audio_spikes=aud_spikes[:, t])

            # Train Concept B (Optional: Multiple concepts)
            # 混乱を避けるため、まずはAのみを強力に学習させるデモにするか、
            # あるいはAとBを交互に学習させるか。
            # ここでは「Aの連合」に集中するためAのみ学習。

        self.is_trained = True
        self._record_targets()  # 学習後のリザーバ状態を新たなターゲットとして更新
        return f"✅ Training Completed ({epochs} Epochs). Association A-A formed."

    def infer(self, vis_amp_A: float, aud_amp_A: float, vis_amp_B: float, aud_amp_B: float) -> Tuple[plt.Figure, Dict[str, float]]:
        """
        推論実行。入力強度に応じて混合入力を生成し、リザーバの応答を可視化。
        """
        self.lac.eval()
        self.lac.reset_state()

        # Input Mixing
        # Input = amp * Pattern
        mixed_vis = (self.vis_A * vis_amp_A +
                     self.vis_B * vis_amp_B).clamp(0, 1)
        mixed_aud = (self.aud_A * aud_amp_A +
                     self.aud_B * aud_amp_B).clamp(0, 1)

        vis_spikes = self.encoder.encode(mixed_vis, 'image', 'rate')
        aud_spikes = self.encoder.encode(mixed_aud, 'audio', 'rate')

        history = []
        for t in range(self.time_steps):
            # 入力がない場合はNoneを渡す (Encoderは0スパイクを生成するが、LAC側でNoneチェックしてもよい)
            # ここでは0スパイクを渡す
            v_in = vis_spikes[:, t] if (
                vis_amp_A > 0 or vis_amp_B > 0) else None
            a_in = aud_spikes[:, t] if (
                aud_amp_A > 0 or aud_amp_B > 0) else None

            out = self.lac(visual_spikes=v_in, audio_spikes=a_in)
            history.append(out.detach())

        activity_tensor = torch.stack(history).squeeze(1)  # (Time, Neurons)
        mean_activity = activity_tensor.mean(dim=0)

        # Similarity Check
        sim_A = F.cosine_similarity(mean_activity.unsqueeze(
            0), self.target_A_activity.unsqueeze(0)).item()
        sim_B = F.cosine_similarity(mean_activity.unsqueeze(
            0), self.target_B_activity.unsqueeze(0)).item()

        # Visualization
        fig = plt.figure(figsize=(10, 6))

        # 1. Raster Plot
        ax1 = plt.subplot(2, 1, 1)
        heatmap = ax1.imshow(activity_tensor.T.numpy(
        ), aspect='auto', interpolation='nearest', cmap='inferno', vmin=0, vmax=1)
        ax1.set_title("Liquid Association Cortex Activity (Spike Raster)")
        ax1.set_ylabel("Neurons")
        ax1.set_xlabel("Time Step")
        plt.colorbar(heatmap, ax=ax1)

        # 2. Concept Activation
        ax2 = plt.subplot(2, 1, 2)
        concepts = ['Concept A (Bell)', 'Concept B (Whistle)']
        scores = [sim_A, sim_B]
        colors = ['blue', 'green']

        bars = ax2.bar(concepts, scores, color=colors)
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Concept Recall Strength (Cosine Similarity)")
        ax2.axhline(y=0.0, color='k', linestyle='-')

        # 値を表示
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        return fig, {"Concept A": sim_A, "Concept B": sim_B}

# --- Gradio UI Setup ---


brain = BrainSystem()


def train_brain(epochs):
    return brain.train(int(epochs))


def run_inference(vis_A, aud_A, vis_B, aud_B):
    return brain.infer(vis_A, aud_A, vis_B, aud_B)


with gr.Blocks(title="SNN Unified Perception Demo") as demo:
    gr.Markdown("# 🧠 SNN Unified Perception: 'Hearing Colors' Demo")
    gr.Markdown("""
    **Phase 9-6**: 液状連合野 (Liquid Association Cortex) による共感覚的想起のデモ。
    
    1. **Train**: [Train Association] ボタンを押して、脳に「視覚Aと音声A」の結びつきを学習させます。
    2. **Test**: 音声スライダーだけを上げてみてください。視覚入力がゼロでも、「Concept A」が想起される（ニューロンが発火し類似度が上がる）様子が観察できます。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Training Phase")
            epochs_slider = gr.Slider(
                minimum=5, maximum=50, value=30, step=5, label="Training Epochs")
            train_btn = gr.Button(
                "🔗 Train Association (Visual A + Audio A)", variant="primary")
            train_status = gr.Textbox(label="Status", value="Not trained.")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Sensory Input Control")
            gr.Markdown("入力強度を調整して脳の反応を見てみましょう。")

            with gr.Row():
                vis_A_slider = gr.Slider(
                    0, 1, value=0, label="👁️ Visual A (Bell)")
                aud_A_slider = gr.Slider(
                    0, 1, value=0, label="👂 Audio A (Bell Sound)")

            with gr.Row():
                vis_B_slider = gr.Slider(
                    0, 1, value=0, label="👁️ Visual B (Whistle)")
                aud_B_slider = gr.Slider(
                    0, 1, value=0, label="👂 Audio B (Whistle Sound)")

            infer_btn = gr.Button("🧠 Process Inputs", variant="secondary")

    with gr.Row():
        plot_output = gr.Plot(label="Brain Activity State")
        label_output = gr.Label(label="Detected Concept")

    # Event Handlers
    train_btn.click(train_brain, inputs=[
                    epochs_slider], outputs=[train_status])

    # リアルタイム性を高めるため、スライダー変更時にも発火させる設定
    inputs = [vis_A_slider, aud_A_slider, vis_B_slider, aud_B_slider]
    outputs = [plot_output, label_output]

    infer_btn.click(run_inference, inputs=inputs, outputs=outputs)
    # スライダーを動かして即座に見たい場合は以下を有効化 (重い場合はOFF)
    # for inp in inputs:
    #     inp.change(run_inference, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
