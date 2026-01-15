# scripts/demos/brain/run_improved_captioning.py
# ファイルパス: scripts/demos/brain/run_improved_captioning.py
# 日本語タイトル: 概念キャプション生成 (精度向上版)
# 説明: パラメータを調整し、SNNが画像から正確な概念を抽出できるか再挑戦する。

import os
import sys
import torch
import torch.nn.functional as F
import logging
import time
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# プロジェクトルート設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from snn_research.models.hybrid.concept_spikformer import ConceptSpikformer
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedCaptioning")

def log(msg):
    print(f"[SmartBrain] {msg}")
    sys.stdout.flush()

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def train_brain_intensive(model, bridge, loader, device, epochs=15):
    """
    概念獲得のための集中学習。
    概念整合性損失(Concept Loss)の重みを高く設定。
    """
    trainer = ConceptAugmentedTrainer(
        model=model, 
        bridge=bridge, 
        learning_rate=0.002, 
        concept_loss_weight=10.0, # [重要] 概念の一致を分類タスクより10倍重視する
        device=device
    )
    
    # 学習率スケジューラ（手動適用）
    scheduler = StepLR(trainer.optimizer, step_size=5, gamma=0.5)
    
    model.train()
    log(f"Starting intensive training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        concept_loss_sum = 0
        batches = 0
        
        for imgs, concepts_batch, labels in loader:
            primary_concepts = list(concepts_batch[0])
            loss_dict = trainer.train_step(imgs, primary_concepts, labels)
            
            total_loss += loss_dict["total_loss"]
            concept_loss_sum += loss_dict["concept_loss"]
            batches += 1
            
        scheduler.step()
        avg_concept_loss = concept_loss_sum / batches
        
        # コンセプト損失が十分下がっているか確認
        log(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/batches:.4f} | Concept Loss: {avg_concept_loss:.4f}")

def generate_captions(model, bridge, test_loader, device, all_concepts_list):
    model.eval()
    
    # 概念辞書の構築
    log("Building Concept Dictionary...")
    concept_library = {}
    with torch.no_grad():
        for concept in all_concepts_list:
            c_spike = bridge.symbol_to_spike(concept, batch_size=1).to(device)
            c_rep = model.forward_conceptual(c_spike)
            
            # [修正] 辞書登録時にProjectionを通す
            c_proj = model.get_concept_projection(c_rep)
            
            c_proj = F.normalize(c_proj, p=2, dim=1)
            concept_library[concept] = c_proj

    log("\n>>> Brain Visual Interpretation Test <<<")
    
    count = 0
    correct_concept_retrieval = 0
    total_checks = 0
    
    with torch.no_grad():
        for imgs, true_concepts_batch, labels in test_loader:
            if count >= 10: break # 10枚テスト
            imgs = imgs.to(device)
            
            # 視覚思考ベクトル生成
            _ = model(imgs) 
            visual_thoughts = model.get_internal_state()
            visual_thoughts = F.normalize(visual_thoughts, p=2, dim=1)
            
            for i in range(len(imgs)):
                if count >= 10: break
                
                label = labels[i].item()
                thought_vec = visual_thoughts[i].unsqueeze(0)
                
                # 類似度計算
                scores = []
                for c_text, c_vec in concept_library.items():
                    sim = torch.mm(thought_vec, c_vec.t()).item()
                    scores.append((c_text, sim))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                top_3 = scores[:3]
                
                # 正解概念リストの構築
                ground_truth = []
                for col in true_concepts_batch:
                    if i < len(col): ground_truth.append(col[i])
                
                # 評価: Top3の中に正解概念が含まれているか？
                top_concepts = [c for c, s in top_3]
                hit = any(c in ground_truth for c in top_concepts)
                if hit: correct_concept_retrieval += 1
                total_checks += 1
                
                print(f"\n[Image: Digit {label}]")
                print(f"  GT Concepts : {ground_truth}")
                print(f"  Brain Thinks: {top_3}")
                print(f"  Judgment    : {'✅ Understands' if hit else '❌ Confused'}")
                
                count += 1
    
    print(f"\nScore: {correct_concept_retrieval}/{total_checks} images correctly conceptualized.")

def main():
    device = get_device()
    log(f"Device: {device}")

    # データセット（学習データを増やして汎化性能を上げる）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # 学習用 5000枚 (前回3000枚から増量)
        train_subset = Subset(mnist_data, range(5000))
        # テスト用
        test_subset = Subset(mnist_data, range(5000, 5100))
    except:
        return

    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(train_subset, concept_map)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = ConceptAugmentedDataset(test_subset, concept_map)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # モデル構築
    all_concepts = set()
    for c_list in concept_map.values():
        all_concepts.update(c_list)
    all_concepts.add("unknown")
    all_concepts_list = list(all_concepts)
    
    concept_dim = 128
    bridge = NeuroSymbolicBridge(embed_dim=concept_dim, concepts=all_concepts_list).to(device)
    
    model = ConceptSpikformer(
        img_size=28, patch_size=4, in_channels=1,
        embed_dim=128, concept_dim=concept_dim,
        num_classes=10
    ).to(device)

    # 実行
    train_brain_intensive(model, bridge, train_loader, device, epochs=15)
    generate_captions(model, bridge, test_loader, device, all_concepts_list)

if __name__ == "__main__":
    main()