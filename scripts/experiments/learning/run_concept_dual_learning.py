# scripts/experiments/learning/run_concept_dual_learning.py
# ファイルパス: scripts/experiments/learning/run_concept_dual_learning.py
# 日本語タイトル: 概念・具体同時学習デモ (Concept-Concrete Dual Learning)
# 修正: AdaptiveLIFNeuronのパラメータ名を修正 (v_threshold -> base_threshold)。

import os
import sys
import torch
import torch.nn as nn
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts
from snn_research.models.bio.visual_cortex import VisualCortex

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConceptLearning")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class SimpleConceptBrain(nn.Module):
    """
    実験用の簡易脳モデル。
    視覚野(VisualCortex)と概念野(Linear Projection)を持つ。
    """
    def __init__(self, num_classes=10, embed_dim=64):
        super().__init__()
        # 1. 視覚野 (Bottom-up)
        # 修正: neuron_paramsのキーを base_threshold に変更
        self.visual_cortex = VisualCortex(
            in_channels=1, 
            base_channels=16, 
            time_steps=4,
            neuron_params={'base_threshold': 1.0, 'v_reset': 0.0} 
        )
        
        # 視覚野の出力次元 (VisualCortexの仕様に合わせる: base_channels*8)
        visual_out_dim = 16 * 8 
        
        # 2. 統合層 (Integration)
        self.integrator = nn.Linear(visual_out_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. 概念入力の射影 (Top-down)
        # Bridgeからのスパイク(概念ベクトル)を受け取る
        self.concept_proj = nn.Linear(embed_dim, embed_dim)
        
        # 4. 分類ヘッド (Task Output)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 内部状態保持用
        self.last_internal_state = None

    def forward_sensory(self, x):
        """感覚入力の処理"""
        # VisualCortexは (Batch, Time, Features) を返す
        features = self.visual_cortex(x)
        # 時間平均をとる
        features_mean = features.mean(dim=1)
        # 統合層へ
        return self.norm(self.integrator(features_mean))

    def forward_conceptual(self, concept_spikes):
        """概念入力の処理"""
        return self.concept_proj(concept_spikes)

    def integrate(self, sensory_rep, conceptual_rep):
        """
        感覚表現と概念表現を統合する。
        """
        # 概念ヒントがある場合は加算
        if conceptual_rep is not None:
            integrated = sensory_rep + 0.3 * conceptual_rep
        else:
            integrated = sensory_rep
            
        self.last_internal_state = integrated
        return self.classifier(integrated)

    def get_internal_state(self):
        """概念整合性損失の計算に使用する内部状態を返す"""
        return self.last_internal_state
        
    def forward(self, x):
        """通常推論用（概念入力なし）"""
        sensory = self.forward_sensory(x)
        return self.classifier(sensory)

def main():
    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. データセットの準備 (MNIST + 概念タグ)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # ダウンロードとロード
    try:
        mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    except Exception as e:
        logger.error(f"Failed to load MNIST: {e}")
        return

    # 動作確認用にデータを小さくサブセット化 (最初の500枚)
    subset_indices = range(500)
    mnist_subset = Subset(mnist_data, subset_indices)
    
    # 概念マッピングの作成と適用
    concept_map = create_mnist_concepts()
    train_dataset = ConceptAugmentedDataset(mnist_subset, concept_map)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 2. コンポーネントの構築
    # Neuro-Symbolic Bridge
    all_concepts = set()
    for concepts in concept_map.values():
        all_concepts.update(concepts)
    all_concepts.add("unknown")
    
    bridge = NeuroSymbolicBridge(
        embed_dim=64, 
        concepts=list(all_concepts)
    ).to(device)
    
    # モデル
    model = SimpleConceptBrain(num_classes=10, embed_dim=64).to(device)
    
    # トレーナー
    trainer = ConceptAugmentedTrainer(
        model=model,
        bridge=bridge,
        learning_rate=0.005,
        concept_loss_weight=1.0, 
        device=device
    )

    # 3. 学習ループ
    epochs = 5
    logger.info("Starting Dual-Stream Learning (Concrete + Abstract)...")
    
    try:
        for epoch in range(epochs):
            total_loss = 0
            task_loss_sum = 0
            concept_loss_sum = 0
            batches = 0
            
            for imgs, concepts_batch, labels in train_loader:
                # concepts_batch は column-major のタプル形式になっているため
                # 各サンプルの「最初の概念」を取り出してリスト化する
                # concepts_batch[0] はバッチ内の全サンプルの1つ目の概念のタプル
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
            
            logger.info(f"Epoch {epoch+1}/{epochs} | "
                        f"Total Loss: {avg_total:.4f} | "
                        f"Task(Tree): {avg_task:.4f} | "
                        f"Concept(Forest): {avg_concept:.4f}")

        logger.info("Training Complete. The model has learned both specific features and abstract concepts.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()