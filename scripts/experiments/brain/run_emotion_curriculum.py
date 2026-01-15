# scripts/experiments/brain/run_emotion_curriculum.py
# 目的: 3段階のカリキュラム学習により、「高い基礎能力」と「自律的な感情学習」を両立させる。
#       Stage 1: Cortexのみ学習 (Task Loss) -> 視覚能力の獲得
#       Stage 2: Amygdalaのみ学習 (Value Loss) -> 正しい価値観の形成
#       Stage 3: Cortexを学習 (Value Pursuit) -> 自律的な成長

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
    print(f"Running Emotion Curriculum Learning on {device}...")

    # データセット
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # データ量: 5000枚
    subset = Subset(mnist_data, range(5000))
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    # 脳モデル
    brain = EmotionalConceptBrain(num_classes=10).to(device)
    
    # ==========================================
    # Stage 1: Vision Pre-training (Cortex Only)
    # ==========================================
    print("\n>>> Stage 1: Vision Pre-training (Acquiring Sight) <<<")
    # Amygdalaは関係ないので凍結せずとも、Lossに加えなければ更新されないが
    # 明示的にCortexのパラメータだけをOptimizerに渡す
    optimizer_cortex = torch.optim.Adam(brain.cortex.parameters(), lr=0.002)
    
    for epoch in range(10): # 10エポックで基礎を作る
        brain.train()
        total_acc = 0
        total_loss = 0
        total_count = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 順伝播
            logits, _ = brain(imgs)
            
            # Task Lossのみ！
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            optimizer_cortex.zero_grad()
            loss.backward()
            optimizer_cortex.step()
            
            # 精度計測
            preds = logits.argmax(dim=1)
            total_acc += (preds == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Task Acc = {total_acc/total_count:.2%} | Loss = {total_loss/len(loader):.4f}")

    # ==========================================
    # Stage 2: Value Alignment (Amygdala Only)
    # ==========================================
    print("\n>>> Stage 2: Value Alignment (Forming Values based on Intelligence) <<<")
    # Cortexを固定し、Amygdalaに「今の脳の状態が良いか悪いか」を教え込む
    for param in brain.cortex.parameters():
        param.requires_grad = False
        
    optimizer_amygdala = torch.optim.Adam(brain.amygdala.parameters(), lr=0.002)
    
    for epoch in range(5):
        total_val_loss = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Cortexは推論モード（Dropoutなし等）で固定的な特徴を出させる
            brain.cortex.eval()
            brain.amygdala.train()
            
            # 順伝播
            logits, emotion = brain(imgs)
            
            # 正解かどうか判定
            preds = logits.argmax(dim=1)
            is_correct = (preds == labels).float().unsqueeze(1)
            
            # 正解なら+1.0 (快), 不正解なら-1.0 (不快)
            target_value = is_correct * 2.0 - 1.0
            
            # Value Lossのみ
            loss = nn.MSELoss()(emotion, target_value)
            
            optimizer_amygdala.zero_grad()
            loss.backward()
            optimizer_amygdala.step()
            
            total_val_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Amygdala Calibration Loss = {total_val_loss/len(loader):.4f}")

    # ==========================================
    # Stage 3: Value Pursuit (Autonomous Learning)
    # ==========================================
    print("\n>>> Stage 3: Autonomous Value Pursuit (Labels Removed!) <<<")
    # 今度はAmygdalaを固定し、Cortexを「Amygdalaが喜ぶ方向」へ更新する
    # これは「正解ラベル」を使わない自律学習
    
    for param in brain.amygdala.parameters():
        param.requires_grad = False
    for param in brain.cortex.parameters():
        param.requires_grad = True # Cortexの固定解除
        
    # 微調整なので学習率は下げる
    optimizer_pursuit = torch.optim.Adam(brain.cortex.parameters(), lr=0.0005)
    
    brain.train()
    brain.amygdala.eval() # 価値基準はブレないように
    
    for epoch in range(10):
        emotion_stats = []
        
        for imgs, _ in loader: # ラベルは無視！
            imgs = imgs.to(device)
            
            logits, emotion = brain(imgs)
            emotion_stats.extend(emotion.detach().cpu().numpy().flatten())
            
            # Loss = -Emotion (感情最大化)
            # エントロピー項を入れて、迷いを減らす（決断させる）
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
            
            loss = -emotion.mean() + 0.05 * entropy
            
            optimizer_pursuit.zero_grad()
            loss.backward()
            optimizer_pursuit.step()
            
        avg_emo = np.mean(emotion_stats)
        max_emo = np.max(emotion_stats)
        
        print(f"Epoch {epoch+1}: Emotion State [Avg: {avg_emo:.4f}, Max: {max_emo:.4f}]")
        
    print("Done. Ideally, Avg Emotion should increase, implying the brain is self-correcting.")

if __name__ == "__main__":
    main()