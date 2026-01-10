# ファイルパス: scripts/demos/brain/run_conscious_broadcast_demo.py
# 日本語タイトル: Conscious Broadcast Demo (Global Workspace Theory)
# 目的・内容:
#   脳内の異なるモジュール（視覚、思考、恒常性）が意識の座を巡って競合する様子をシミュレーションする。
#   通常は「視覚」が優位だが、緊急時（空腹や痛み）には「恒常性」が割り込んで意識をジャックすることを確認する。

from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
import os
import sys
import torch
import logging
import time

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)


def run_consciousness_demo():
    print("""
    ============================================================
       👁️ CONSCIOUS BROADCAST DEMO (Global Workspace) 👁️
    ============================================================
    """)

    device = "cpu"
    dim = 64

    # 1. Initialize Global Workspace
    gwt = GlobalWorkspace(dim=dim).to(device)

    # 2. Simulate Modules
    # 各モジュールは信号(Tensor)を出力する
    logger.info("🧠 Initializing Brain Modules: [Vision], [Thought], [Body]...")

    # シナリオ:
    # 平穏な状態 -> 何かを見る -> 考え事をする -> 突然の腹痛(緊急)

    for step in range(15):
        inputs = {}

        # --- Module 1: Vision (視覚) ---
        # 常に環境情報を送ってくる
        vision_signal = torch.randn(1, dim).to(device) * 1.0  # 通常強度
        if 2 <= step <= 5:
            # 興味深いものを見た！
            vision_signal *= 3.0
            logger.debug(f"Step {step}: Vision is excited!")
        inputs["Vision 📷"] = vision_signal

        # --- Module 2: Thought (思考/言語) ---
        # ランダムな思考
        thought_signal = torch.randn(1, dim).to(device) * 0.8
        if 6 <= step <= 9:
            # 深い思考モード
            thought_signal *= 4.0
            logger.debug(f"Step {step}: Deep thought...")
        inputs["Thought 💭"] = thought_signal

        # --- Module 3: Body (身体/恒常性) ---
        # 通常は静かだが...
        body_signal = torch.randn(1, dim).to(device) * 0.2
        if step >= 11:
            # 緊急事態！ (痛みや空腹)
            body_signal *= 10.0  # 圧倒的強度
            logger.debug(f"Step {step}: BODY EMERGENCY!")
        inputs["Body 💓"] = body_signal

        # --- GWT Step ---
        # 意識による選択と放送
        result = gwt(inputs)

        winner = result["winner"]
        # broadcast_vec = result["broadcast"]
        # salience = result["salience"]

        # Visualize
        # 簡易バーチャートでAttentionを表示
        # attn_str = " | ".join(
        #     [f"{k}: {salience[i]:.2f}" for i, k in enumerate(inputs.keys())])

        icon = ""
        if "Vision" in winner:
            icon = "👀 Seeing"
        elif "Thought" in winner:
            icon = "🤔 Thinking"
        elif "Body" in winner:
            icon = "😫 Feeling"

        logger.info(f"Step {step:02d}: Winner -> [{winner}] {icon}")
        # logger.info(f"   Attn: {attn_str}")

        time.sleep(0.1)  # 読みやすくするため少し待機

    logger.info("\n✅ Demo Result Analysis:")
    logger.info("   1. Steps 0-1: Random fluctuations (Mind wandering)")
    logger.info(
        "   2. Steps 2-5: Vision dominates (Attention captured by scene)")
    logger.info(
        "   3. Steps 6-9: Thought dominates (Internal simulation/planning)")
    logger.info(
        "   4. Steps 11+: Body interrupts everything (Survival instinct)")

    logger.info("🎉 Global Workspace Demo Completed.")


if __name__ == "__main__":
    run_consciousness_demo()
