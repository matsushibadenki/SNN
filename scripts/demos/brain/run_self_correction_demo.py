# ファイルパス: scripts/demos/brain/run_self_correction_demo.py
# 日本語タイトル: Self-Correction Demo v1.1 (Eureka Test)
# 目的・内容:
#   [Update] 改良されたメタ認知モジュールを使用し、学習停滞からの脱却（Eureka）を確認する。

from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging

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


# 簡易タスク: y = 2x の学習

class SimpleLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.linear.weight, 0.1)  # Bad init

    def forward(self, x):
        return self.linear(x)


def run_self_correction():
    print("""
    =========================================================
       🧠 SELF-CORRECTION DEMO (Phase 3: Eureka Mode) 🧠
    =========================================================
    """)

    device = "cpu"

    # 1. Initialize Systems
    learner = SimpleLearner().to(device)
    base_lr = 0.001  # Extremely low learning rate
    optimizer = optim.SGD(learner.parameters(), lr=base_lr)
    criterion = nn.MSELoss()

    # Sensitivityを高めに設定して早めにキレさせる
    meta_brain = MetaCognitiveSNN(config={"sensitivity": 0.15}).to(device)

    logger.info(f"🎓 Learner initialized with LOW learning rate ({base_lr}).")
    logger.info("   Target Task: Learn y = 2x (Current weight ~0.1)")

    # 2. Simulation Loop
    target_weight = 2.0

    # Allow slightly more steps for frustration to build up
    for step in range(40):
        # Data
        x = torch.randn(1, 1).to(device)
        y_target = x * target_weight

        # Forward
        y_pred = learner(x)
        loss = criterion(y_pred, y_target)
        current_error = loss.item()

        # --- Metacognition Step ---
        meta_state = meta_brain.monitor({"error": current_error})

        # Adaptive Control
        dynamic_lr = base_lr * meta_state["focus_level"]

        # Apply Dynamic LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = dynamic_lr

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Visualization ---
        current_w = learner.linear.weight.item()

        status_icon = "😐"
        if meta_state["focus_level"] >= 50.0:
            status_icon = "🤯 EUREKA!"
        elif meta_state["frustration"] > 0.8:
            status_icon = "😡 Rage"
        elif meta_state["frustration"] > 0.5:
            status_icon = "😠 Annoyed"
        if current_error < 0.1:
            status_icon = "😃 Happy"

        if step % 2 == 0 or meta_state["focus_level"] >= 50.0:
            logger.info(
                f"Step {step:02d}: Err={current_error:.4f} | "
                f"W={current_w:.2f} | "
                f"Frust={meta_state['frustration']:.2f} | "
                f"Focus={meta_state['focus_level']:.1f}x | "
                f"LR={dynamic_lr:.4f} {status_icon}"
            )

        # Success Check
        if current_error < 0.01 and abs(current_w - target_weight) < 0.1:
            logger.info(
                f"✨ PROBLEM SOLVED at step {step}! (Weight: {current_w:.4f})")
            break

    # Final check
    final_w = learner.linear.weight.item()
    logger.info(f"\n🏁 Final Weight: {final_w:.4f}")

    if abs(final_w - target_weight) < 0.2:
        logger.info(
            "✅ Self-Correction Successful: 'Eureka Mode' broke through the stagnation.")
    else:
        logger.warning("❌ Correction Failed: Still stuck.")


if __name__ == "__main__":
    run_self_correction()
