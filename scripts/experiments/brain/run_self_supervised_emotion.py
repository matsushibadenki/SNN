# scripts/experiments/brain/run_self_supervised_emotion.py
# 修正: "Value Pursuit" アプローチへの転換。
#       Phase 1で価値観を確立し、Phase 2でその価値(Emotion)を最大化するように脳を更新する。

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from snn_research.models.hybrid.emotional_concept_brain import EmotionalConceptBrain

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    print(f"Running Emotion-Driven Learning (Value Pursuit) on {device}...")

    # データセット
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # データ量を増やして基礎を固める (3000 -> 5000)
    subset = Subset(mnist_data, range(5000))
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    # 脳モデル
    brain = EmotionalConceptBrain(num_classes=10).to(device)
    
    # Phase 1用のオプティマイザ (全体を学習)
    optimizer_p1 = torch.optim.Adam(brain.parameters(), lr=0.002)
    
    print("\n>>> Phase 1: Pre-training (Forming Values & Skills) <<<")
    # ここで躓くとPhase2に行けないので、しっかり学習させる
    epochs_p1 = 15
    
    for epoch in range(epochs_p1):
        brain.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits, emotion = brain(imgs) 
            
            # 外部評価
            preds = logits.argmax(dim=1)
            is_correct = (preds == labels).float().unsqueeze(1) 
            correct += is_correct.sum().item()
            total += labels.size(0)

            # ターゲット: 正解なら 1.0 (快), 不正解なら -1.0 (不快)
            target_value = is_correct * 2.0 - 1.0 
            
            # 損失: 価値観のズレ + タスクのズレ
            # Amygdalaに「何が良い状態か」を教え込む
            value_loss = nn.MSELoss()(emotion, target_value)
            task_loss = nn.CrossEntropyLoss()(logits, labels)
            
            loss = value_loss + task_loss
            
            optimizer_p1.zero_grad()
            loss.backward()
            optimizer_p1.step()
            
            total_loss += loss.item()
            
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs_p1}: Task Acc = {acc:.2f} | Total Loss = {total_loss/len(loader):.4f}")

    print("\n>>> Phase 2: Autonomous Value Pursuit (No Labels!) <<<")
    # ここからはラベルを使わない (Unsupervised)
    # 「Amygdalaが快と感じる状態」を目指してCortexだけを更新する
    
    # 1. Amygdala（価値基準）を固定する。
    #    そうしないと「何でもかんでも快と感じる」ように価値観自体が堕落してしまうため。
    for param in brain.amygdala.parameters():
        param.requires_grad = False
        
    # 2. Cortexのみを学習するオプティマイザ
    optimizer_p2 = torch.optim.Adam(brain.cortex.parameters(), lr=0.0005) # 微調整なので学習率低め
    
    brain.train()
    
    for epoch in range(10):
        emotion_values = []
        
        for imgs, _ in loader: # ラベルは無視！
            imgs = imgs.to(device)
            
            # 順伝播
            logits, emotion = brain(imgs)
            
            emotion_values.extend(emotion.detach().cpu().numpy().flatten())
            
            # 【重要】イリヤ・サツケバーの仮説の実装
            # 価値関数(Emotion)を最大化するように、脳の活動(Logits/Internal State)を更新する
            # Loss = -Emotion (Emotionが高いほどLossが下がる)
            
            # ただし、単にEmotionを上げるだけだと、特定の「安全なパターンの絵」を幻視し始める可能性があるため
            # エントロピー最小化（決断すること）も補助的に入れる
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
            
            # Goal: Feel Good + Be Decisive
            loss = -emotion.mean() + 0.1 * entropy
            
            optimizer_p2.zero_grad()
            loss.backward()
            optimizer_p2.step()
            
        avg_emo = np.mean(emotion_values)
        max_emo = np.max(emotion_values)
        
        print(f"Epoch {epoch+1}: Emotion State [Avg: {avg_emo:.4f}, Max: {max_emo:.4f}]")
        
        if avg_emo > 0.0:
            print("   -> The brain is starting to feel confident autonomously!")
        
    print("Done. Autonomous learning complete.")

if __name__ == "__main__":
    main()