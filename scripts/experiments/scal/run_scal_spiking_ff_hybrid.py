# ファイルパス: scripts/runners/run_scal_spiking_ff_hybrid.py
# Title: run_scal_spiking_ff_hybrid
# Description:


from snn_research.core.ensemble_scal import EnsembleSCAL
from snn_research.training.trainers.forward_forward import ForwardForwardTrainer, SpikingForwardForwardLayer
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../..')))


class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=3, device='cpu'):
        super().__init__()
        self.scal = EnsembleSCAL(
            in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        # SNNへの入力用に正規化 (0-1範囲に収めるのが理想)
        self.norm = nn.LayerNorm(out_features).to(device)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return self.norm(out['output'])  # [B, F]

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print(f"Pre-training SCAL for {epochs} epochs...")
        for epoch in range(epochs):
            for data, _ in data_loader:
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)
        print("SCAL Pre-training Complete.")


class HybridSpikingFFTrainer(ForwardForwardTrainer):
    """
    SCAL + Spiking FF のハイブリッドトレーナー
    """

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
        y_fake = (y + torch.randint(1, self.num_classes,
                  (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        return x_pos, x_neg

    def predict(self, data_loader: DataLoader, noise_level: float = 0.0) -> float:
        self.model.eval()
        # SNNは内部状態(膜電位)を持つため、推論時はリセットが必要だが
        # forward内で毎回初期化しているので問題なし

        # デバイス転送
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

                features = self.feature_extractor(data)
                features = features.to(self.device)

                class_goodness = torch.zeros(
                    data.size(0), self.num_classes, device=self.device)

                for label_idx in range(self.num_classes):
                    temp_labels = torch.full(
                        (data.size(0),), label_idx, dtype=torch.long, device=self.device)
                    h = self.overlay_y_on_x(features, temp_labels)

                    for layer in self.execution_pipeline:
                        # Spiking Layer
                        if isinstance(layer, SpikingForwardForwardLayer):
                            h = layer.forward(h)  # Returns spikes [B, T, F]

                            # Goodness = Mean Firing Rate Squared
                            # 時間(1)と特徴(2)の次元で平均
                            dims = list(range(1, h.dim()))
                            g = h.mean(dim=dims).pow(2)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)

                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100.0 * correct / total


def run_scal_spiking_ff():
    print("=== SCAL + Spiking FF Hybrid Experiment (The Bio-Brain) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../../../../data')
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    pretrain_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 1. SCAL Pre-training
    print("\n--- Phase 1: Pre-training SCAL (Visual Cortex) ---")
    scal_dim = 1000
    scal_extractor = SCALFeatureExtractor(
        in_features=784, out_features=scal_dim, n_models=3, device=device)
    scal_extractor.fit(pretrain_loader, epochs=3)

    # 2. Spiking Neural Network
    print("\n--- Phase 2: Training Spiking FF Layers (Bio-Brain) ---")

    # SNN構造: SCAL出力(1000) -> 隠れ層 -> 出力
    # ReLUは不要 (TrainerがLIF化)
    snn_model = nn.Sequential(
        nn.Linear(scal_dim, 1500),
        nn.Linear(1500, 1500),
        nn.Linear(1500, 1500)
    ).to(device)

    config = {
        "use_snn": True,          # 重要: SNNモードON
        "time_steps": 25,         # 時間分解能を少し上げる
        "learning_rate": 0.005,
        "ff_threshold": 0.08,     # 低めの発火率を目指す
        "num_epochs": 10
    }

    trainer = HybridSpikingFFTrainer(
        feature_extractor=scal_extractor,
        model=snn_model,
        device=device,
        config=config,
        num_classes=10
    )

    # Training
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Validating Epoch {epoch}...")
        acc = trainer.predict(test_loader)
        print(
            f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f} - SNN Accuracy: {acc:.2f}%")

    # 3. Robustness
    print("\n--- Phase 3: Noise Robustness (SNN) ---")
    noise_levels = [0.0, 0.4, 0.6, 0.8]
    for noise in noise_levels:
        acc = trainer.predict(test_loader, noise_level=noise)
        print(f"Noise {noise:<3}: {acc:.2f}%")

    print("\nExperiment Finished.")


if __name__ == "__main__":
    run_scal_spiking_ff()
