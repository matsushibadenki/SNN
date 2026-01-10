# ファイルパス: scripts/demos/social/run_social_cognition_demo.py
# 日本語タイトル: Social Cognition Demo (Reading Intentions)
# 目的・内容:
#   2体のエージェント（Actor, Observer）を用いた社会性デモ。
#   Actorが目的地に向かって移動する様子を、ObserverがToMモジュールを使って観察し、
#   移動完了前に目的地（意図）を言い当てる能力を検証する。

import os
import sys
import torch
import torch.nn as nn
import logging
import time
import numpy as np

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

from snn_research.social.theory_of_mind import TheoryOfMindEncoder  # noqa: E402


class ActorAgent:
    """目的地に向かって移動する単純なエージェント"""

    def __init__(self, start_pos, target_pos):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.target = np.array(target_pos, dtype=np.float32)
        self.speed = 0.1
        self.history = []

    def step(self):
        # Move towards target
        direction = self.target - self.pos
        dist = np.linalg.norm(direction)
        if dist > self.speed:
            move = (direction / dist) * self.speed
            self.pos += move
        else:
            self.pos = self.target.copy()

        # Record history (x, y)
        self.history.append(self.pos.copy())
        if len(self.history) > 16:  # Keep last 16 steps
            self.history.pop(0)

    def get_trajectory(self):
        # Pad if history is short
        traj = np.array(self.history)
        if len(traj) < 16:
            pad = np.zeros((16 - len(traj), 2)) + traj[0]  # Pad with start pos
            traj = np.vstack([pad, traj])
        # [1, 16, 2]
        return torch.tensor(traj, dtype=torch.float32).unsqueeze(0)


def run_social_demo():
    print("""
    =======================================================
       🤝 SOCIAL COGNITION DEMO (Theory of Mind) 🤝
    =======================================================
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚙️ Running on {device.upper()}")

    # 1. Initialize ToM Engine (Observer)
    # Input: [x, y] coordinates sequence
    # Output: [x, y] predicted target
    tom_engine = TheoryOfMindEncoder(
        input_dim=2,
        hidden_dim=64,
        intent_dim=2,
        model_type="mamba"  # Use the fast Mamba core
    ).to(device)

    optimizer = torch.optim.AdamW(tom_engine.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    logger.info("🧠 Observer Agent (ToM) initialized.")

    # 2. Training Phase (Interaction / Observation)
    logger.info("🎓 Phase 1: Observing & Learning Intentions...")

    for episode in range(50):
        # Scenario: Actor moves from (0,0) to random target in (1,1) square
        target = np.random.rand(2)
        actor = ActorAgent(start_pos=[0, 0], target_pos=target)

        # Actor acts for 20 steps
        for t in range(20):
            actor.step()

            # Observer watches (needs at least some history)
            if t > 5:
                traj = actor.get_trajectory().to(device)
                target_tensor = torch.tensor(
                    target, dtype=torch.float32).to(device).unsqueeze(0)

                # Predict
                tom_engine.train()
                optimizer.zero_grad()
                pred_target = tom_engine(traj)

                loss = criterion(pred_target, target_tensor)
                loss.backward()
                optimizer.step()

        if (episode+1) % 10 == 0:
            logger.info(
                f"   Episode {episode+1}: Prediction Loss = {loss.item():.4f}")

    # 3. Testing Phase (Real-time Prediction)
    logger.info("\n🔮 Phase 2: Real-time Intent Prediction Test")

    # New Scenario
    real_target = [0.8, 0.2]  # Goal: Bottom Right
    actor = ActorAgent(start_pos=[0, 0], target_pos=real_target)

    logger.info(f"   Actor's Secret Goal: {real_target}")

    for t in range(15):
        actor.step()

        # Observer predicts every step
        traj = actor.get_trajectory().to(device)

        start_time = time.time()
        with torch.no_grad():
            pred = tom_engine.predict_goal(traj)
        lat = (time.time() - start_time) * 1000

        pred_pos = pred.cpu().numpy()[0]
        dist = np.linalg.norm(pred_pos - real_target)

        # Display
        status = "🤔 Guessing..."
        if dist < 0.1:
            status = "💡 I KNOW!"

        logger.info(
            f"   Step {t:02d}: Pos={actor.pos.round(2)} -> Predicted Goal={pred_pos.round(2)} | Err={dist:.2f} | {status} ({lat:.2f}ms)")

        if dist < 0.1:
            logger.info("   ✅ Correctly predicted intent before arrival!")
            break

    logger.info("🎉 Social Cognition Demo Completed.")


if __name__ == "__main__":
    run_social_demo()
