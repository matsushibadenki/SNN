# scripts/experiments/brain/run_emotion_curriculum_v2.py
# 日本語タイトル: 感情駆動型カリキュラム学習 v2 (検証付き)
# 目的: 1. 高い基礎能力(Vision)を獲得させる。
#       2. 正しい価値観(Amygdala)を形成する。
#       3. ラベルなし自律学習(Value Pursuit)を行い、その効果をテストデータで検証する。

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

def evaluate_accuracy(model, loader, device):
    """モデルの現在の正解率を計測する（学習はしない）"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    print(f"Running Emotion Curriculum Learning v2 on {device}...")

    # データセット (MNIST Full)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # データ量を増やして基礎を固める (10000枚)
    train_subset = Subset(full_data, range(10000))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 脳モデル
    brain = EmotionalConceptBrain(num_classes=10).to(device)
    
    # ==========================================
    # Stage 1: Vision Pre-training (Cortex Only)
    # ==========================================
    print("\n>>> Stage 1: Vision Pre-training (Acquiring Sight) <<<")
    # 学習率調整
    optimizer_cortex = torch.optim.Adam(brain.cortex.parameters(), lr=0.001)
    
    # 80%を超えるまで、あるいは最大20エポックやる
    for epoch in range(20):
        brain.cortex.train()
        total_loss = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = brain(imgs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            optimizer_cortex.zero_grad()
            loss.backward()
            optimizer_cortex.step()
            total_loss += loss.item()
            
        # 精度チェック
        val_acc = evaluate_accuracy(brain, test_loader, device)
        print(f"Epoch {epoch+1}: Test Acc = {val_acc:.2%} | Loss = {total_loss/len(train_loader):.4f}")
        
        if val_acc > 0.85: # 85%超えたら十分
            print("-> Vision acquired! Moving to next stage.")
            break

    # ==========================================
    # Stage 2: Value Alignment (Amygdala Only)
    # ==========================================
    print("\n>>> Stage 2: Value Alignment (Forming Values) <<<")
    # Cortex固定
    for param in brain.cortex.parameters():
        param.requires_grad = False
    
    optimizer_amygdala = torch.optim.Adam(brain.amygdala.parameters(), lr=0.002)
    
    for epoch in range(5):
        brain.amygdala.train()
        total_val_loss = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 順伝播
            logits, emotion = brain(imgs)
            
            # 正解判定
            preds = logits.argmax(dim=1)
            is_correct = (preds == labels).float().unsqueeze(1)
            
            # 正解なら+1.0, 不正解なら-1.0
            target_value = is_correct * 2.0 - 1.0
            
            loss = nn.MSELoss()(emotion, target_value)
            
            optimizer_amygdala.zero_grad()
            loss.backward()
            optimizer_amygdala.step()
            total_val_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Amygdala Loss = {total_val_loss/len(train_loader):.4f}")

    # ==========================================
    # Stage 3: Value Pursuit (Autonomous Learning)
    # ==========================================
    print("\n>>> Stage 3: Autonomous Value Pursuit (NO LABELS!) <<<")
    print("Hypothesis: Improving 'Emotion' autonomously will improve 'Test Accuracy'.")
    
    # Amygdala固定、Cortex学習再開
    for param in brain.amygdala.parameters():
        param.requires_grad = False
    for param in brain.cortex.parameters():
        param.requires_grad = True
        
    optimizer_pursuit = torch.optim.Adam(brain.cortex.parameters(), lr=0.0001) # 慎重に更新
    
    initial_acc = evaluate_accuracy(brain, test_loader, device)
    print(f"Initial Test Acc: {initial_acc:.2%}")
    
    brain.train()
    brain.amygdala.eval()
    
    for epoch in range(10):
        emotion_stats = []
        
        for imgs, _ in train_loader: # ラベルは見ない！
            imgs = imgs.to(device)
            
            logits, emotion = brain(imgs)
            emotion_stats.extend(emotion.detach().cpu().numpy().flatten())
            
            # 価値最大化 (Maximize Emotion)
            # エントロピー正則化で「自信を持った判断」を促す
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
            
            # Loss = -Emotion + entropy_penalty
            loss = -emotion.mean() + 0.1 * entropy
            
            optimizer_pursuit.zero_grad()
            loss.backward()
            optimizer_pursuit.step()
            
        # 評価
        avg_emo = np.mean(emotion_stats)
        current_acc = evaluate_accuracy(brain, test_loader, device)
        
        print(f"Epoch {epoch+1}: Emotion Avg = {avg_emo:.4f} | Test Acc = {current_acc:.2%} (Delta: {current_acc - initial_acc:+.2%})")
        
        if current_acc > initial_acc:
            print("   -> 🌟 EUREKA! Autonomous improvement confirmed!")

    print("Experiment Complete.")

if __name__ == "__main__":
    main()