# scripts/demos/brain/run_concept_dialogue.py
# ファイルパス: scripts/demos/brain/run_concept_dialogue.py
# 日本語タイトル: 概念対話デモ (Concept Dialogue Interface)
# 機能説明: 学習済みConceptSpikformerを用いて、ユーザーと「画像の印象」について対話する。
#           SNNは「なぜそう思ったか」を、Attention Mapや類似概念を使って説明しようと試みる。

import os
import sys
import torch
import torch.nn.functional as F
import logging
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset

# プロジェクトルート設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from snn_research.models.hybrid.concept_spikformer import ConceptSpikformer
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
from snn_research.io.concept_dataset import ConceptAugmentedDataset, create_mnist_concepts

# ログ設定（対話の邪魔にならないよう抑制）
logging.getLogger().setLevel(logging.WARNING)

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

class ConceptBrainInterface:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.bridge = None
        self.concept_library = {}
        self.mnist_data = None
        self.concept_map = None

    def initialize_and_train(self, epochs=5): # デモ用に短く
        print("\n🧠 Initializing Concept Brain...")
        
        # データセット
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_subset = Subset(full_data, range(3000)) # 高速学習用
        self.mnist_data = Subset(full_data, range(3000, 3100)) # テスト用
        
        self.concept_map = create_mnist_concepts()
        dataset = ConceptAugmentedDataset(train_subset, self.concept_map)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 概念リスト
        all_concepts = set()
        for c_list in self.concept_map.values():
            all_concepts.update(c_list)
        all_concepts.add("unknown")
        all_concepts_list = list(all_concepts)
        
        # モデル構築
        concept_dim = 128
        self.bridge = NeuroSymbolicBridge(embed_dim=concept_dim, concepts=all_concepts_list).to(self.device)
        self.model = ConceptSpikformer(
            img_size=28, patch_size=4, in_channels=1,
            embed_dim=128, concept_dim=concept_dim, num_classes=10
        ).to(self.device)
        
        # トレーナー (v3 Prototype)
        trainer = ConceptAugmentedTrainer(
            model=self.model, bridge=self.bridge, 
            learning_rate=0.002, concept_loss_weight=1.0, device=self.device
        )
        
        print(f"📚 Studying concepts for {epochs} epochs (Quick Learning)...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for imgs, concepts_batch, labels in loader:
                primary = list(concepts_batch[0])
                loss = trainer.train_step(imgs, primary, labels)["total_loss"]
                total_loss += loss
            print(f"   - Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")
            
        print("✨ Brain is ready!\n")
        self._build_concept_library(all_concepts_list)

    def _build_concept_library(self, concepts):
        self.model.eval()
        with torch.no_grad():
            for c in concepts:
                c_spike = self.bridge.symbol_to_spike(c, batch_size=1).to(self.device)
                c_rep = self.model.forward_conceptual(c_spike)
                if hasattr(self.model, 'get_concept_projection'):
                    c_rep = self.model.get_concept_projection(c_rep)
                c_rep = F.normalize(c_rep, p=2, dim=1)
                self.concept_library[c] = c_rep

    def see_and_think(self, index=None):
        if index is None:
            index = random.randint(0, len(self.mnist_data)-1)
        
        img, label = self.mnist_data[index]
        img_input = img.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img_input)
            thought = self.model.get_internal_state()
            thought = F.normalize(thought, p=2, dim=1)
            
            # 概念検索
            scores = []
            for c_text, c_vec in self.concept_library.items():
                sim = torch.mm(thought, c_vec.t()).item()
                scores.append((c_text, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_concepts = scores[:3]
            
            return label, top_concepts

def main():
    device = get_device()
    brain = ConceptBrainInterface(device)
    brain.initialize_and_train(epochs=8) # デモなので8エポック
    
    print("--- 🤖 Concept Brain Dialogue ---")
    print("Commands: [n]ext image, [q]uit")
    
    while True:
        cmd = input("\nAction? (n/q): ").strip().lower()
        if cmd == 'q':
            break
        
        if cmd == 'n' or cmd == '':
            label, thoughts = brain.see_and_think()
            print(f"\n👁️  I see an image. (Ground Truth: Digit {label})")
            print(f"🧠 My impression:")
            for i, (concept, score) in enumerate(thoughts):
                confidence = int(score * 100)
                bar = "█" * (confidence // 5)
                print(f"   {i+1}. {concept:<15} [{bar:<20}] ({confidence}%)")
                
            top_concept = thoughts[0][0]
            print(f"\n🗣️  Agent: 'It looks like a {top_concept} structure to me.'")

if __name__ == "__main__":
    main()