# scripts/visualization/visualize_concept_structure.py
# ファイルパス: scripts/visualization/visualize_concept_structure.py
# 日本語タイトル: 概念空間の構造可視化 (Concept Space Visualization)
# 機能説明: ConceptBrainを学習させ、画像（木）と概念（森）が埋め込み空間でどのように配置されたかを図示する。

from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.models.experimental.concept_brain import ConceptBrain
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# プロジェクトルート設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


# 出力を確実にするためのprintラッパー

def log(msg):
    print(f"[ConceptVis] {msg}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # 1. データ準備 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        full_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform)
    except Exception as e:
        log(f"Error loading MNIST: {e}")
        return

    # 学習用（少量で高速に）
    train_indices = range(1000)
    train_subset = Subset(full_dataset, train_indices)
    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(train_subset, concept_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. モデル構築
    all_concepts = set()
    for concepts in concept_map.values():
        all_concepts.update(concepts)
    all_concepts.add("unknown")

    # 概念ブリッジ
    bridge = NeuroSymbolicBridge(
        embed_dim=64,
        concepts=list(all_concepts)
    ).to(device)

    # 脳モデル
    model = ConceptBrain(num_classes=10, embed_dim=64).to(device)

    # トレーナー
    trainer = ConceptAugmentedTrainer(
        model=model,
        bridge=bridge,
        learning_rate=0.005,
        concept_loss_weight=1.0,
        device=device
    )

    # 3. 学習実行
    epochs = 5
    log("Starting training...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, concepts_batch, labels in train_loader:
            primary_concepts = list(concepts_batch[0])  # 各サンプルの代表概念

            loss_dict = trainer.train_step(
                specific_data=imgs,
                abstract_concepts=primary_concepts,
                targets=labels
            )
            total_loss += loss_dict["total_loss"]

        log(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    log("Training complete. Extracting internal representations...")

    # 4. 内部表現の抽出と可視化
    model.eval()
    embeddings = []
    labels_list = []

    # テスト用データ（別の500枚）
    test_indices = range(1000, 1500)
    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            # 画像から内部表現を取得（概念入力なし＝純粋なボトムアップ認識）
            reps = model.forward_sensory(imgs)
            embeddings.append(reps.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels_list, axis=0)

    # PCAで2次元圧縮
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # プロット
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(
        'Concept Space Visualization: Trees (Images) aligned by Forest (Concepts)')
    plt.xlabel('PC1 (Concept Axis 1)')
    plt.ylabel('PC2 (Concept Axis 2)')
    plt.grid(True, alpha=0.3)

    # 概念の意味合いを注釈として追加（概念マップの情報を一部表示）
    # クラスごとの重心を計算
    for digit in range(10):
        center = X_pca[y == digit].mean(axis=0)
        # 代表的な概念タグを1つ取得
        concept_tag = concept_map[digit][0]
        plt.text(center[0], center[1], f"{digit}:{concept_tag}",
                 fontsize=9, weight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    output_path = "workspace/concept_space_pca.png"
    plt.savefig(output_path)
    log(f"Visualization saved to {output_path}")
    log("Check the image to see if 'conceptually similar' digits are clustered together.")


if __name__ == "__main__":
    main()
