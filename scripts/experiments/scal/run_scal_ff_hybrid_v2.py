# ファイルパス: scripts/runners/run_scal_ff_hybrid_v2.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer, ForwardForwardLayer
from snn_research.core.ensemble_scal import EnsembleSCAL

class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=3, device='cpu'):
        super().__init__()
        self.scal = EnsembleSCAL(in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        # 信号レベルを整えるための正規化層を追加
        self.norm = nn.LayerNorm(out_features).to(device)
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        # SCALの生の出力を正規化して返す
        return self.norm(out['output'])

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print(f"Pre-training SCAL for {epochs} epochs...")
        for epoch in range(epochs):
            # プログレスバーなしで高速に回す
            for data, _ in data_loader:
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)
        print("SCAL Pre-training Complete.")

class HybridFFTrainer(ForwardForwardTrainer):
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def generate_negative_data(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = features.to(self.device)
            
        x_pos = self.overlay_y_on_x(features, y)
        y_fake = (y + torch.randint(1, self.num_classes, (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        return x_pos, x_neg

    def predict(self, data_loader: DataLoader, noise_level: float = 0.0) -> float:
        self.model.eval()
        self.model.to(self.device)
        for layer in self.execution_pipeline:
            layer.to(self.device)
            if hasattr(layer, 'block'):
                layer.block.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if noise_level > 0:
                    noise = torch.randn_like(data) * noise_level
                    data = data + noise
                
                batch_size = data.size(0)
                features = self.feature_extractor(data)
                features = features.to(self.device)
                
                class_goodness = torch.zeros(batch_size, self.num_classes, device=self.device)
                
                for label_idx in range(self.num_classes):
                    temp_labels = torch.full((batch_size,), label_idx, dtype=torch.long, device=self.device)
                    h = self.overlay_y_on_x(features, temp_labels)
                    
                    for layer in self.execution_pipeline:
                        if isinstance(layer, ForwardForwardLayer):
                            h = layer.forward(h)
                            dims = list(range(1, h.dim()))
                            g = h.pow(2).mean(dim=dims)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)
                
                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total

def run_optimized_scal_ff():
    print("=== Optimized SCAL-FF Hybrid Experiment (V2) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../../../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    pretrain_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 1. SCAL (Visual Cortex)
    print("\n--- Phase 1: Pre-training SCAL (5 Epochs + Normalization) ---")
    scal_out_dim = 1000 
    scal_extractor = SCALFeatureExtractor(in_features=784, out_features=scal_out_dim, n_models=3, device=device)
    
    # 修正: 十分な学習時間を確保
    scal_extractor.fit(pretrain_loader, epochs=5)

    # 2. FF Layers
    print("\n--- Phase 2: Training FF Layers ---")
    
    # モデルサイズを少し調整
    ff_model = nn.Sequential(
        nn.Linear(scal_out_dim, 1500), 
        nn.ReLU(),
        nn.Linear(1500, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1500),
        nn.ReLU()
    ).to(device)

    config = {
        "learning_rate": 0.03,
        "ff_threshold": 2.0,
        "num_epochs": 10 # エポック数を戻す
    }
    
    trainer = HybridFFTrainer(
        feature_extractor=scal_extractor,
        model=ff_model,
        device=device,
        config=config,
        num_classes=10
    )

    # Training Loop
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        # 毎エポックの全件検証は重いのでスキップしても良いが、推移を見るために表示
        if epoch % 2 == 0:
            print(f"Validating Epoch {epoch}...")
            acc = trainer.predict(test_loader)
            print(f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f} - Accuracy: {acc:.2f}%")
        else:
            print(f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f}")
        
    save_path = os.path.join(os.path.dirname(__file__), '../../../../results/checkpoints/scal_ff_v2.pth')
    trainer.save_checkpoint(save_path)

    # 3. Robustness Test
    print("\n--- Phase 3: Noise Robustness Evaluation ---")
    noise_levels = [0.0, 0.4, 0.6, 0.8]
    print(f"{'Noise Level':<15} | {'Accuracy':<10}")
    print("-" * 30)
    
    for noise in noise_levels:
        acc = trainer.predict(test_loader, noise_level=noise)
        print(f"{noise:<15.1f} | {acc:.2f}%")

    print("\nExperiment Finished.")

if __name__ == "__main__":
    run_optimized_scal_ff()