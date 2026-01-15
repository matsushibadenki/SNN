# scripts/visualization/visualize_concept_attention.py
# ファイルパス: scripts/visualization/visualize_concept_attention.py
# 日本語タイトル: 概念誘導注意の可視化 (Visualizing Concept-Guided Attention)
# 機能説明: 画像（数字の0など）に対し、異なる概念（Curve, Void等）を入力した際に、
#           モデルのAttentionがどこに反応するかをヒートマップで出力する。

from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.models.hybrid.concept_spikformer import ConceptSpikformer
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# プロジェクトルート設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def log(msg):
    print(f"[AttnVis] {msg}")


def visualize_attention(model, bridge, img_tensor, concepts, device, save_path):
    """
    1枚の画像に対し、指定された概念リストそれぞれでAttentionMapを生成して保存する。
    """
    model.eval()
    bridge.eval()

    # 画像の前処理
    # img_tensor: (1, 28, 28)
    img_input = img_tensor.unsqueeze(0).to(device)  # (1, 1, 28, 28)

    # 視覚特徴の抽出 (Bottom-up)
    with torch.no_grad():
        visual_feats = model.forward_sensory(img_input)

    # 可視化の準備
    num_concepts = len(concepts)
    fig, axes = plt.subplots(
        1, num_concepts + 1, figsize=(4 * (num_concepts + 1), 4))

    # 元画像の表示
    original_img = img_tensor.squeeze().cpu().numpy()
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 各概念ごとのAttentionを表示
    for i, concept_text in enumerate(concepts):
        # 概念の埋め込み (Top-down)
        with torch.no_grad():
            concept_spike = bridge.symbol_to_spike(
                concept_text, batch_size=1).to(device)
            concept_rep = model.forward_conceptual(concept_spike)

            # 統合 (Integration) -> ここでAttentionが計算される
            _ = model.integrate(visual_feats, concept_rep)

            # Attention Mapの取得
            # shape: (Batch, Heads, N_patches, M_concepts) -> (1, 4, 49, 1)
            attn_map = model.get_last_attention_map()

        if attn_map is None:
            continue

        # ヘッド平均をとる
        attn_avg = attn_map.mean(dim=1).squeeze()  # (49,)

        # パッチサイズに合わせて2Dに戻す (7x7) -> Spikformer(28x28, patch=4)の場合
        grid_size = int(np.sqrt(attn_avg.shape[0]))
        attn_2d = attn_avg.view(grid_size, grid_size).cpu().numpy()

        # 元画像サイズ(28x28)に拡大
        attn_resized = cv2.resize(
            attn_2d, (28, 28), interpolation=cv2.INTER_CUBIC)

        # ヒートマップ表示
        ax = axes[i + 1]
        ax.imshow(original_img, cmap='gray', alpha=0.5)
        im = ax.imshow(attn_resized, cmap='jet', alpha=0.6)  # 重ね合わせ
        ax.set_title(f"Focus: '{concept_text}'")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log(f"Saved attention map to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    log(f"Using device: {device}")

    # 1. 簡易学習 (Attentionを形成するため)
    log("Preparing dataset and model...")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        mnist_data = datasets.MNIST(
            './data', train=True, download=True, transform=transform)
        # 学習は高速化のため1000枚程度で行う
        train_subset = Subset(mnist_data, range(1000))
    except:
        return

    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(train_subset, concept_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Bridge
    all_concepts = set()
    for c_list in concept_map.values():
        all_concepts.update(c_list)
    all_concepts.add("unknown")

    concept_dim = 128
    bridge = NeuroSymbolicBridge(
        embed_dim=concept_dim, concepts=list(all_concepts)).to(device)

    # Model
    model = ConceptSpikformer(
        img_size=28, patch_size=4, in_channels=1,
        embed_dim=128, concept_dim=concept_dim,
        num_classes=10
    ).to(device)

    trainer = ConceptAugmentedTrainer(
        model=model, bridge=bridge, learning_rate=0.002, device=device)

    # Short Training Loop
    log("Running short training to form concept associations...")
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for imgs, concepts_batch, labels in train_loader:
            primary_concepts = list(concepts_batch[0])
            loss = trainer.train_step(imgs, primary_concepts, labels)[
                "total_loss"]
            total_loss += loss
        log(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    # 2. Attentionの可視化実験
    log("Generating Concept-Guided Attention Maps...")

    # テスト画像: 数字の「0」を取得 (クラス0の最初の画像を探す)
    target_img = None
    for img, label in mnist_data:
        if label == 0:
            target_img = img
            break

    if target_img is not None:
        # 実験: 「0」の画像に対して、異なる概念でどこを見るか試す
        # "round": 丸い部分全体を見るはず
        # "void": 中央の空白を見るかもしれない
        # "curve": 曲線部分
        # "straight": (0には無いため) 混乱するか、反応なしかも
        test_concepts = ["round", "void", "curve", "straight"]

        visualize_attention(
            model, bridge, target_img, test_concepts,
            device, "workspace/concept_attention_0.png"
        )

    # テスト画像: 数字の「1」
    target_img_1 = None
    for img, label in mnist_data:
        if label == 1:
            target_img_1 = img
            break

    if target_img_1 is not None:
        test_concepts_1 = ["vertical", "straight", "curve", "round"]
        visualize_attention(
            model, bridge, target_img_1, test_concepts_1,
            device, "workspace/concept_attention_1.png"
        )

    log("Done. Check 'workspace/concept_attention_*.png'.")


if __name__ == "__main__":
    main()
