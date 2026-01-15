# scripts/experiments/brain/run_emotional_action.py
# 日本語タイトル: 感情駆動型身体性エージェント (Emotional Embodied Agent)
# 目的: 「視覚→感情→行動」のループを構築する。
#       外部からの明示的な指示（教師あり分類）ではなく、
#       「痛い/美味しい」という身体的経験を通じて、数字の意味（毒/食料）を学習し、行動を変容させる。

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

# 以前作成した安定版脳モデルを使用
from snn_research.models.hybrid.emotional_concept_brain import EmotionalConceptBrain

class MotorCortex(nn.Module):
    """
    感情と認識に基づいて行動を決定する運動野。
    Action: [0: Avoid/Discard, 1: Approach/Eat]
    """
    def __init__(self, input_dim=128+1, hidden_dim=64):
        super().__init__()
        # 入力: 脳の内部表現(128) + 感情値(1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 2択のアクション
        )
    
    def forward(self, cortex_state, emotion_value):
        # 感情を行動のバイアスとして強く使う
        # cortex_state: (B, 128)
        # emotion_value: (B, 1)
        combined = torch.cat([cortex_state, emotion_value], dim=1)
        logits = self.network(combined)
        return logits

class EmotionalAgent(nn.Module):
    def __init__(self, brain):
        super().__init__()
        self.brain = brain
        # 運動野を追加
        self.motor = MotorCortex()
        
    def act(self, img):
        # 1. 認知 & 感情 (Brain)
        # ここでは分類ロジット(logits)ではなく、内部表現(sensory_vec)を使いたいが、
        # EmotionalConceptBrainのインターフェース上、logitsが返る。
        # 内部状態を取得する。
        _ = self.brain(img)
        internal_state = self.brain.get_internal_state() # (B, 128)
        
        # 感情は、Amygdalaから現在の入力をどう感じているかを取得
        # (forward内で計算されているが、ここでもう一度取得するか、forwardの戻り値を使う)
        # 簡易的に再計算する（コストは低い）
        # 本来は脳のforwardが (prediction, emotion) を返す
        logits, emotion = self.brain(img)
        
        # 2. 行動選択 (Motor)
        action_logits = self.motor(internal_state, emotion)
        
        return action_logits, emotion

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    print(f"Running Emotional Embodiment Experiment on {device}...")

    # --- 1. 環境設定 (Survival Environment) ---
    # 0-4: Food (Reward +1.0), 5-9: Poison (Reward -1.0)
    def get_environment_feedback(label, action):
        is_food = (label < 5)
        # Action 0: Avoid, 1: Eat
        
        if action == 1: # 食べた
            if is_food: return 1.0, "Yummy! (+1)"
            else:       return -1.0, "Ouch! Poison! (-1)"
        else: # 避けた
            # 避けた場合は無害だが、お腹は満たされない（0.0 または 僅かなペナルティ）
            if is_food: return -0.1, "Missed food... (-0.1)"
            else:       return 0.1, "Safe choice. (+0.1)"

    # --- 2. エージェント構築 ---
    # 学習済みの脳があればロードするのがベストだが、ここでは新規作成して高速学習させる
    brain = EmotionalConceptBrain(num_classes=10).to(device)
    # 視覚野(CNN)だけは事前に軽く学習済みとして、重みを初期化（本来は前回の重みを使う）
    # ここでは実験時間を短縮するため、未学習の状態から「痛い目を見て覚える」過程を見る
    
    agent = EmotionalAgent(brain).to(device)
    
    # オプティマイザ
    # 脳(Amygdala含む)と運動野を同時に更新
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    # データセット
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(Subset(mnist_data, range(2000)), batch_size=1, shuffle=True) # 1つずつ体験する

    print("\n>>> Survival Simulation Start <<<")
    print("Agent is exploring the world. (0-4: Food, 5-9: Poison)")
    
    survival_score = 0
    history = []
    
    agent.train()
    
    for step, (img, label) in enumerate(loader):
        img, label = img.to(device), label.item()
        
        # --- 1. 感知と行動 ---
        action_logits, emotion = agent.act(img)
        
        # 確率的行動選択 (Exploration)
        probs = F.softmax(action_logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        
        # --- 2. 環境からのフィードバック ---
        reward, msg = get_environment_feedback(label, action)
        
        # --- 3. 学習 (Experience) ---
        # A. 価値観の更新 (Amygdala): 「この画像＝この報酬」を覚える
        #    食べた結果痛かったら、その画像は「不快」と紐づく
        target_val = torch.tensor([[reward]], device=device)
        loss_amygdala = nn.MSELoss()(emotion, target_val)
        
        # B. 行動の更新 (Motor): 報酬が得られる行動を強化 (Reinforcement Learning / Policy Gradient)
        #    log_prob * reward (報酬最大化) -> loss = -log_prob * reward
        log_prob = torch.log(probs[0, action])
        loss_motor = -log_prob * reward
        
        # C. 視覚の更新: 価値判断に役立つ特徴を抽出するように
        loss_total = loss_amygdala + loss_motor
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # --- ログ ---
        survival_score += reward
        history.append(reward)
        if len(history) > 100: history.pop(0)
        recent_avg = sum(history) / len(history)
        
        if step % 100 == 0:
            emotion_val = emotion.item()
            # 感情と行動の一致度を見る
            # 感情がポジティブなら食べる(1)、ネガティブなら避ける(0)はず
            feeling = "Good" if emotion_val > 0 else "Bad"
            did = "Ate" if action == 1 else "Avoided"
            
            print(f"Step {step:4d} | Digit: {label} | Feel: {emotion_val:+.2f} ({feeling}) -> Did: {did} | Res: {msg}")
            print(f"           | Recent Score Avg: {recent_avg:+.2f} | Total Score: {survival_score:.1f}")

        if step >= 1000: break

    print("\n>>> Simulation Complete <<<")
    if recent_avg > 0.5:
        print("RESULT: The agent successfully learned to survive based on emotional values!")
    else:
        print("RESULT: The agent is struggling. More experience needed.")

if __name__ == "__main__":
    main()