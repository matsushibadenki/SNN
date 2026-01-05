# ファイルパス: scripts/runners/run_scal_spiking_ff_fashion.py
# Title: run_scal_spiking_ff_fashion
# Description:

from snn_research.core.ensemble_scal import EnsembleSCAL
from snn_research.training.trainers.forward_forward import ForwardForwardTrainer, SpikingForwardForwardLayer
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import random
from collections import deque

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../..')))


class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=3, device='cpu'):
        super().__init__()
        # Fashion-MNISTは複雑なので、SCALのモデル数(n_models)を増やして表現力を上げる
        self.scal = EnsembleSCAL(
            in_features=in_features, out_features=out_features, n_models=5).to(device)
        self.device = device
        self.norm = nn.LayerNorm(out_features).to(device)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return self.norm(out['output'])

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print(f"Pre-training SCAL for {epochs} epochs...")
        for epoch in range(epochs):
            for data, _ in data_loader:
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)
        print("SCAL Pre-training Complete.")


class HippocampalReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, features, label):
        for i in range(features.size(0)):
            self.buffer.append(
                (features[i].detach().clone(), label[i].detach().clone()))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        features, labels = zip(*batch)
        return torch.stack(features), torch.stack(labels)

    def __len__(self):
        return len(self.buffer)


class BioReplayTrainer(ForwardForwardTrainer):
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.replay_buffer = HippocampalReplayBuffer(capacity=5000)

    def generate_negative_data(self, x: torch.Tensor, y: torch.Tensor, features=None) -> tuple[torch.Tensor, torch.Tensor]:
        if features is None:
            with torch.no_grad():
                features = self.feature_extractor(x)
            features = features.to(self.device)

        if self.model.training:
            self.replay_buffer.push(features, y)

        x_pos = self.overlay_y_on_x(features, y)
        y_fake = (y + torch.randint(1, self.num_classes,
                  (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        return x_pos, x_neg

    def train_sleep_phase(self, batch_size=64, noise_intensity=1.0, replay_ratio=0.5):
        for layer in self.execution_pipeline:
            layer.train()

        noise_size = int(batch_size * (1 - replay_ratio))
        replay_size = batch_size - noise_size

        noise_input = torch.randn(noise_size, 1000).to(
            self.device) * noise_intensity
        dummy_labels = torch.randint(
            0, self.num_classes, (noise_size,)).to(self.device)
        x_noise = self.overlay_y_on_x(noise_input, dummy_labels)

        if len(self.replay_buffer) > replay_size:
            features_mem, labels_mem = self.replay_buffer.sample(replay_size)
            features_mem, labels_mem = features_mem.to(
                self.device), labels_mem.to(self.device)
            x_mem_pos = self.overlay_y_on_x(features_mem, labels_mem)
            y_mem_fake = (labels_mem + torch.randint(1, self.num_classes,
                          (len(labels_mem),)).to(self.device)) % self.num_classes
            x_mem_neg = self.overlay_y_on_x(features_mem, y_mem_fake)
        else:
            return 0.0

        total_loss = 0.0

        # Unlearning (Noise)
        x_curr = x_noise
        for layer in self.execution_pipeline:
            if isinstance(layer, SpikingForwardForwardLayer):
                layer.optimizer.zero_grad()
                spikes = layer.forward(x_curr)
                dims = list(range(1, spikes.dim()))
                g = spikes.mean(dim=dims).pow(2)
                loss = torch.log(1 + torch.exp(g - layer.threshold)).mean()
                loss.backward()
                layer.optimizer.step()
                with torch.no_grad():
                    x_curr = layer.forward(x_curr).detach()
            else:
                x_curr = layer(x_curr)

        # Consolidation (Replay)
        x_p, x_n = x_mem_pos, x_mem_neg
        for layer in self.execution_pipeline:
            if isinstance(layer, SpikingForwardForwardLayer):
                l_loss, _, _ = layer.train_step(x_p, x_n)
                total_loss += l_loss
                with torch.no_grad():
                    x_p = layer.forward(x_p).detach()
                    x_n = layer.forward(x_n).detach()
            else:
                x_p = layer(x_p)
                x_n = layer(x_n)

        return total_loss

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

                features = self.feature_extractor(data)
                features = features.to(self.device)

                class_goodness = torch.zeros(
                    data.size(0), self.num_classes, device=self.device)

                for label_idx in range(self.num_classes):
                    temp_labels = torch.full(
                        (data.size(0),), label_idx, dtype=torch.long, device=self.device)
                    h = self.overlay_y_on_x(features, temp_labels)

                    for layer in self.execution_pipeline:
                        if isinstance(layer, SpikingForwardForwardLayer):
                            h = layer.forward(h)
                            dims = list(range(1, h.dim()))
                            g = h.mean(dim=dims).pow(2)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)

                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total


def run_scal_spiking_ff_fashion():
    print("=== SCAL + Spiking FF + Replay [Fashion-MNIST] ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Data: Fashion-MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNISTの平均・標準偏差
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../../../../data')
    # Datasetを変更
    train_dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    pretrain_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 1. SCAL Pre-training
    print("\n--- Phase 1: Pre-training SCAL (Visual Cortex) ---")
    scal_dim = 1000
    # モデル数を増やして複雑なパターンに対応
    scal_extractor = SCALFeatureExtractor(
        in_features=784, out_features=scal_dim, n_models=5, device=device)
    scal_extractor.fit(pretrain_loader, epochs=5)

    # 2. Spiking Neural Network
    print("\n--- Phase 2: Training Bio-Brain on Fashion Items ---")

    snn_model = nn.Sequential(
        nn.Linear(scal_dim, 2000),  # ニューロン数を増やす
        nn.Linear(2000, 2000),
        nn.Linear(2000, 1500)
    ).to(device)

    config = {
        "use_snn": True,
        "time_steps": 30,         # 時間分解能を上げて詳細な特徴を捉える
        "learning_rate": 0.003,
        "ff_threshold": 0.15,
        "num_epochs": 15
    }

    trainer = BioReplayTrainer(
        feature_extractor=scal_extractor,
        model=snn_model,
        device=device,
        config=config,
        num_classes=10
    )

    # Training
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)

        # Sleep & Replay
        sleep_iters = len(train_loader) // 5
        sleep_loss_accum = 0.0
        for _ in range(sleep_iters):
            sleep_loss_accum += trainer.train_sleep_phase(
                batch_size=64, noise_intensity=1.0, replay_ratio=0.5)

        avg_sleep_loss = sleep_loss_accum / sleep_iters

        if epoch % 2 == 0 or epoch == config['num_epochs']:
            print(f"Validating Epoch {epoch}...")
            acc = trainer.predict(test_loader)
            print(
                f"Epoch {epoch} - Wake Loss: {metrics['train_loss']:.4f} - Sleep Loss: {avg_sleep_loss:.4f} - SNN Acc: {acc:.2f}%")
        else:
            print(
                f"Epoch {epoch} - Wake Loss: {metrics['train_loss']:.4f} - Sleep Loss: {avg_sleep_loss:.4f}")

    save_path = os.path.join(os.path.dirname(
        __file__), '../../../../results/checkpoints/spiking_ff_fashion.pth')
    trainer.save_checkpoint(save_path)

    # 3. Robustness
    print("\n--- Phase 3: Noise Robustness (Fashion-SNN) ---")
    noise_levels = [0.0, 0.4, 0.6, 0.8]
    for noise in noise_levels:
        acc = trainer.predict(test_loader, noise_level=noise)
        print(f"Noise {noise:<3}: {acc:.2f}%")

    print("\nExperiment Finished.")


if __name__ == "__main__":
    run_scal_spiking_ff_fashion()
