# scripts/demos/brain/run_concept_captioning.py
# ファイルパス: scripts/demos/brain/run_concept_captioning.py
# 日本語タイトル: 概念想起・キャプション生成デモ (Concept Retrieval / Image Captioning)
# 機能説明: 画像を入力し、SNNの脳内にある「概念辞書（Bridge）」の中から、
#           その画像に最もふさわしい概念（抽象タグ）を検索して出力する。

import os
import sys
import torch
import torch.nn.functional as F
import logging
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
logger = logging.getLogger("ConceptCaptioning")

def log(msg):
    print(f"[Captioning] {msg}")

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def train_brain(model, bridge, loader, device, epochs=3):
    """思考能力獲得のための短期学習"""
    trainer = ConceptAugmentedTrainer(
        model=model, bridge=bridge, learning_rate=0.002, 
        concept_loss_weight=2.0, # 概念結合を強く学習させる
        device=device
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, concepts_batch, labels in loader:
            primary_concepts = list(concepts_batch[0])
            loss_dict = trainer.train_step(imgs, primary_concepts, labels)
            total_loss += loss_dict["total_loss"]
        log(f"Training Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

def generate_captions(model, bridge, test_loader, device, all_concepts_list):
    """
    画像を見て、脳内の全概念との類似度を計算し、上位の概念を出力する。
    """
    model.eval()
    
    # 全概念の埋め込みベクトルを事前に計算（辞書として機能）
    log("Building Concept Dictionary in Neural Space...")
    concept_library = {}
    with torch.no_grad():
        for concept in all_concepts_list:
            # 概念文字列 -> スパイク/埋め込み
            c_spike = bridge.symbol_to_spike(concept, batch_size=1).to(device)
            # 概念野での表現に変換
            c_rep = model.forward_conceptual(c_spike) # (1, Dim)
            concept_library[concept] = c_rep

    # 画像に対する推論
    log("\n>>> Visual Thought Process (Image -> Concepts) <<<")
    
    # 最初の5バッチ分だけテスト
    count = 0
    with torch.no_grad():
        for imgs, true_concepts_batch, labels in test_loader:
            if count >= 5: break
            imgs = imgs.to(device)
            
            # 1. 視覚入力 -> 内部表現 (Bottom-up)
            # model(imgs) を呼ぶと integrate(sensory, None) が走る
            _ = model(imgs) 
            # 内部状態（概念空間上の座標）を取得
            visual_thoughts = model.get_internal_state() # (Batch, Dim)
            
            # バッチ内の各画像について
            for i in range(len(imgs)):
                if count >= 5: break
                
                label = labels[i].item()
                thought_vec = visual_thoughts[i].unsqueeze(0) # (1, Dim)
                
                # 2. 概念検索 (Ranking)
                # 思考ベクトルと全概念ベクトルの類似度を計算
                scores = []
                for c_text, c_vec in concept_library.items():
                    # Cosine Similarity
                    sim = F.cosine_similarity(thought_vec, c_vec).item()
                    scores.append((c_text, sim))
                
                # スコア順にソート
                scores.sort(key=lambda x: x[1], reverse=True)
                top_3 = scores[:3]
                
                # 正解概念（データセット定義）
                # true_concepts_batch は column-major なので、rowごとのリストにする必要あり
                # ここでは簡易表示のため、concept_mapから再取得するか、デバッグ表示で済ませる
                # データローダーの構造上、true_concepts_batch[0][i], true_concepts_batch[1][i]... となる
                ground_truth = []
                for col in true_concepts_batch:
                    if i < len(col): ground_truth.append(col[i])
                
                # 結果表示
                print(f"\n[Image: Digit {label}]")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Brain Thinks: {top_3}")
                
                # 考察: 
                # もし Brain Thinks に正解タグが含まれていれば、
                # モデルは「具体的な画素」から「抽象的な意味」への翻訳に成功している。
                
                count += 1

def main():
    device = get_device()
    log(f"Using device: {device}")

    # 1. データセット準備
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # 学習用: 3000枚
        train_subset = Subset(mnist_data, range(3000))
        # テスト用: 別の100枚
        test_subset = Subset(mnist_data, range(3000, 3100))
    except:
        return

    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(train_subset, concept_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = ConceptAugmentedDataset(test_subset, concept_map)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # 2. モデル構築
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

    # 3. 学習（概念獲得）
    log("Learning concepts from images...")
    train_brain(model, bridge, train_loader, device, epochs=3)

    # 4. 推論（キャプション生成）
    generate_captions(model, bridge, test_loader, device, all_concepts_list)

if __name__ == "__main__":
    main()