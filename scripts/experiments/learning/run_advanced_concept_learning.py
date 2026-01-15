# scripts/experiments/learning/run_advanced_concept_learning.py
# ファイルパス: scripts/experiments/learning/run_advanced_concept_learning.py
# 日本語タイトル: 高度概念学習実行スクリプト (Advanced Concept Learning Runner)
# 修正: 概念次元とモデル次元を統一し、Forest(概念)の損失が計算されるように修正。

import os
import sys
import torch
import logging
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

from snn_research.models.hybrid.concept_spikformer import ConceptSpikformer
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts

# 出力用ラッパー
def log(msg):
    print(f"[AdvancedLearning] {msg}")
    sys.stdout.flush()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    log(f"Device selected: {device}")

    # --- 1. Data Preparation ---
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # データ数を増やして学習を安定化 (2000 -> 5000)
        subset_indices = range(5000)
        mnist_subset = Subset(mnist_data, subset_indices)
    except Exception as e:
        log(f"Error loading data: {e}")
        return

    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(mnist_subset, concept_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    log(f"Dataset loaded. Samples: {len(train_dataset)}")

    # --- 2. Concept Bridge Setup ---
    # 全概念の抽出
    all_concepts = set()
    for c_list in concept_map.values():
        all_concepts.update(c_list)
    all_concepts.add("unknown")
    
    # [修正] モデルの内部次元(128)に合わせて概念次元も128にする
    # これにより ConceptAugmentedTrainer での損失計算が可能になる
    concept_dim = 128 
    
    bridge = NeuroSymbolicBridge(
        embed_dim=concept_dim,
        concepts=list(all_concepts)
    ).to(device)
    
    log(f"Neuro-Symbolic Bridge initialized. Vocabulary size: {len(all_concepts)}, Dim: {concept_dim}")

    # --- 3. Advanced Model Setup (ConceptSpikformer) ---
    model = ConceptSpikformer(
        img_size=28,
        patch_size=4,
        in_channels=1,
        embed_dim=128,      # Spikformer内部次元
        concept_dim=concept_dim, # Bridge出力次元 (128で一致させる)
        num_classes=10,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    log("ConceptSpikformer initialized.")

    # --- 4. Trainer Setup ---
    trainer = ConceptAugmentedTrainer(
        model=model,
        bridge=bridge,
        learning_rate=0.002, # 学習率を少し上げる
        concept_loss_weight=1.0, 
        device=device
    )

    # --- 5. Training Loop ---
    epochs = 5
    log(">>> Starting Training (Tree & Forest Dual Learning) <<<")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        task_loss_sum = 0.0
        concept_loss_sum = 0.0
        batches = 0
        
        for imgs, concepts_batch, labels in train_loader:
            primary_concepts = list(concepts_batch[0])
            
            loss_dict = trainer.train_step(
                specific_data=imgs,
                abstract_concepts=primary_concepts,
                targets=labels
            )
            
            total_loss += loss_dict["total_loss"]
            task_loss_sum += loss_dict["task_loss"]
            concept_loss_sum += loss_dict["concept_loss"]
            batches += 1
        
        avg_total = total_loss / batches
        avg_task = task_loss_sum / batches
        avg_concept = concept_loss_sum / batches
        
        elapsed = time.time() - start_time
        log(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
            f"Total: {avg_total:.4f} | "
            f"Tree(Task): {avg_task:.4f} | "
            f"Forest(Concept): {avg_concept:.4f}")

    log(">>> Training Complete <<<")
    log("Check if 'Forest(Concept)' loss is non-zero and decreasing.")

if __name__ == "__main__":
    main()