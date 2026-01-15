# scripts/experiments/brain/run_emotional_action_long.py
# 日本語タイトル: 感情駆動型エージェントの長期生存実験 (Long-term Survival)
# 目的: シミュレーション回数を増やし(3000回)、AIが「生存戦略」を完全にマスターする様子を可視化する。

from snn_research.models.hybrid.emotional_concept_brain import EmotionalConceptBrain
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


class MotorCortex(nn.Module):
    def __init__(self, input_dim=128+1, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, cortex_state, emotion_value):
        combined = torch.cat([cortex_state, emotion_value], dim=1)
        return self.network(combined)


class EmotionalAgent(nn.Module):
    def __init__(self, brain):
        super().__init__()
        self.brain = brain
        self.motor = MotorCortex()

    def act(self, img):
        _ = self.brain(img)
        internal_state = self.brain.get_internal_state()
        _, emotion = self.brain(img)
        action_logits = self.motor(internal_state, emotion)
        return action_logits, emotion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Running Long-term Emotional Embodiment on {device}...")

    # 環境設定
    def get_environment_feedback(label, action):
        is_food = (label < 5)  # 0-4: Food, 5-9: Poison
        if action == 1:  # Eat
            return (1.0, "Yummy!") if is_food else (-1.0, "Ouch!")
        else:  # Avoid
            return (-0.1, "Missed") if is_food else (0.1, "Safe")

    # モデル構築
    brain = EmotionalConceptBrain(num_classes=10).to(device)
    agent = EmotionalAgent(brain).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    # データセット (MNIST)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    # データを増やして長時間シミュレーション
    loader = DataLoader(Subset(mnist_data, range(5000)),
                        batch_size=1, shuffle=True)

    print("\n>>> Survival Simulation Start (3000 Steps) <<<")

    survival_scores = []
    avg_scores = []
    window_size = 100
    recent_rewards = []

    agent.train()

    for step, (img, label) in enumerate(loader):
        if step >= 3000:
            break

        img, label = img.to(device), label.item()

        # 1. 行動選択
        action_logits, emotion = agent.act(img)
        probs = F.softmax(action_logits, dim=1)
        action = torch.multinomial(probs, 1).item()

        # 2. フィードバック
        reward, msg = get_environment_feedback(label, action)

        # 3. 学習
        target_val = torch.tensor([[reward]], device=device)
        loss_amygdala = nn.MSELoss()(emotion, target_val)

        log_prob = torch.log(probs[0, action])
        loss_motor = -log_prob * reward

        loss_total = loss_amygdala + loss_motor

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # ログ記録
        recent_rewards.append(reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)

        avg_score = sum(recent_rewards) / len(recent_rewards)
        survival_scores.append(reward)
        avg_scores.append(avg_score)

        if step % 200 == 0:
            emotion_val = emotion.item()
            feeling = "Good" if emotion_val > 0 else "Bad"
            did = "Eat" if action == 1 else "Avoid"
            print(
                f"Step {step:4d} | Digit {label}: Feel {emotion_val:+.2f} ({feeling}) -> {did} | {msg} | Avg Score: {avg_score:+.2f}")

    # 結果の可視化
    print("\n>>> Generating Survival Curve <<<")
    plt.figure(figsize=(10, 5))
    plt.plot(
        avg_scores, label=f'Moving Avg (Window={window_size})', color='green')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Agent Survival Learning Curve (Emotion-Driven)')
    plt.xlabel('Experience Steps')
    plt.ylabel('Survival Score (Reward)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('workspace/survival_curve_long.png')
    print("Saved 'workspace/survival_curve_long.png'.")

    final_score = avg_scores[-1]
    print(f"Final Survival Score: {final_score:+.2f}")

    if final_score > 0.6:
        print("RESULT: SUCCESS! The agent has mastered survival using its emotions.")
    elif final_score > 0.3:
        print("RESULT: GOOD. The agent is surviving, but makes occasional mistakes.")
    else:
        print("RESULT: STRUGGLING. The environment might be too harsh.")


if __name__ == "__main__":
    main()
