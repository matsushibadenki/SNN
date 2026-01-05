# ファイルパス: scripts/runners/run_continual_learning_demo.py
# Title: run_continual_learning_demo
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
    os.path.dirname(__file__), '../../../')))


# --- 共通コンポーネント ---


class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=5, device='cpu'):
        super().__init__()
        self.scal = EnsembleSCAL(
            in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        self.norm = nn.LayerNorm(out_features).to(device)

    def forward(self, x):
        # MPSクラッシュ対策: メモリを連続化してから転送
        if not x.is_contiguous():
            x = x.contiguous()
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return self.norm(out['output'])

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print(f"  SCAL training ({epochs} epochs)...")
        for _ in range(epochs):
            for data, _ in data_loader:
                if not data.is_contiguous():
                    data = data.contiguous()
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)


class HippocampalReplayBuffer:
    """
    長期記憶バッファ。
    GPUメモリを圧迫しないよう、データはCPUメモリに退避させて保持する。
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, features, label):
        # GPUから切り離し、CPUへ移動して保存
        features_cpu = features.detach().cpu()
        label_cpu = label.detach().cpu()
        for i in range(features.size(0)):
            self.buffer.append((features_cpu[i].clone(), label_cpu[i].clone()))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None
        batch = random.sample(self.buffer, batch_size)
        features, labels = zip(*batch)
        return torch.stack(features), torch.stack(labels)

    def __len__(self): return len(self.buffer)


class ContinualTrainer(ForwardForwardTrainer):
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.replay_buffer = HippocampalReplayBuffer(capacity=10000)

    def generate_negative_data(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.feature_extractor(x)

        # 特徴量を確実にデバイスへ
        features = features.to(self.device)

        # 学習中(Training)かつ覚醒時のみ、経験を海馬バッファに保存
        if self.model.training:
            self.replay_buffer.push(features, y)

        x_pos = self.overlay_y_on_x(features, y)
        y_fake = (y + torch.randint(1, self.num_classes,
                  (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        return x_pos, x_neg

    def train_sleep_phase_with_replay(self, batch_size=64):
        """
        睡眠フェーズ: 海馬バッファから過去の記憶を再生し、学習する。
        """
        for layer in self.execution_pipeline:
            layer.train()

        # バッファからサンプル (CPUにある状態)
        features_mem, labels_mem = self.replay_buffer.sample(batch_size)
        if features_mem is None:
            return 0.0

        # 学習直前にGPUへ転送
        features_mem, labels_mem = features_mem.to(
            self.device), labels_mem.to(self.device)

        # リプレイデータをPositiveとして学習 (Negativeはラベルを偽装して生成)
        x_p = self.overlay_y_on_x(features_mem, labels_mem)
        y_fake = (labels_mem + torch.randint(1, self.num_classes,
                  (len(labels_mem),)).to(self.device)) % self.num_classes
        x_n = self.overlay_y_on_x(features_mem, y_fake)

        total_loss = 0.0
        for layer in self.execution_pipeline:
            if isinstance(layer, SpikingForwardForwardLayer):
                l_loss, _, _ = layer.train_step(x_p, x_n)
                total_loss += l_loss
                # 勾配を切って次の層へ
                with torch.no_grad():
                    x_p = layer.forward(x_p).detach()
                    x_n = layer.forward(x_n).detach()
            else:
                x_p = layer(x_p)
                x_n = layer(x_n)
        return total_loss

    def predict(self, data_loader: DataLoader) -> float:
        """
        特徴抽出を含めた推論パイプライン
        """
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
                data = data.to(self.device)
                target = target.to(self.device)

                # 1. 特徴抽出
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
                            # Goodness = Mean Firing Rate Squared
                            dims = list(range(1, h.dim()))
                            g = h.mean(dim=dims).pow(2)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)

                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100.0 * correct / total


def run_continual_learning_demo():
    print("=== Continual Learning Experiment: MNIST -> Fashion-MNIST ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Data Transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')

    # Load Datasets
    print("Loading Task A: MNIST (Digits)...")
    mnist_train = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, transform=transform)
    loader_a_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
    # 重要: テスト用バッチサイズを削減 (MPSメモリ対策: 1000 -> 100)
    loader_a_test = DataLoader(mnist_test, batch_size=100, shuffle=False)

    print("Loading Task B: Fashion-MNIST (Clothing)...")
    fashion_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(
        data_dir, train=False, transform=transform)
    loader_b_train = DataLoader(fashion_train, batch_size=64, shuffle=True)
    # 重要: テスト用バッチサイズを削減 (MPSメモリ対策: 1000 -> 100)
    loader_b_test = DataLoader(fashion_test, batch_size=100, shuffle=False)

    # Model Setup
    scal_dim = 1000
    # 5つのモデルで多様な特徴を捉える
    scal = SCALFeatureExtractor(
        in_features=784, out_features=scal_dim, n_models=5, device=device)

    snn_model = nn.Sequential(
        nn.Linear(scal_dim, 2000),
        nn.Linear(2000, 2000),
        nn.Linear(2000, 10)
    ).to(device)

    config = {
        "use_snn": True,
        "time_steps": 25,
        "learning_rate": 0.002,
        "ff_threshold": 0.15,
        "num_epochs": 5
    }

    trainer = ContinualTrainer(
        feature_extractor=scal, model=snn_model, device=device, config=config, num_classes=10)

    # --- Task A: MNIST Learning ---
    print("\n[Phase 1] Learning Task A: MNIST")
    scal.fit(loader_a_train, epochs=2)

    for epoch in range(1, 6):
        trainer.train_epoch(loader_a_train, epoch)
        # Sleep & Replay (記憶の定着)
        for _ in range(50):
            trainer.train_sleep_phase_with_replay()

        if epoch % 5 == 0:
            print("  Predicting Task A...")
            acc_a = trainer.predict(loader_a_test)
            print(f"  Epoch {epoch}: MNIST Acc = {acc_a:.2f}%")

    print("Task A Learning Finished. Retaining memory buffer...")

    # --- Task B: Fashion-MNIST Learning ---
    print("\n[Phase 2] Learning Task B: Fashion-MNIST (while remembering MNIST)")
    scal.fit(loader_b_train, epochs=2)

    for epoch in range(1, 6):
        trainer.train_epoch(loader_b_train, epoch)

        # Sleep & Replay (ここでMNISTの記憶とFashionの記憶が混ざって再生される)
        # 睡眠時間を増やして定着を強化
        for _ in range(100):
            trainer.train_sleep_phase_with_replay()

        if epoch % 5 == 0:
            print("  Predicting Task B...")
            acc_b = trainer.predict(loader_b_test)
            print(f"  Epoch {epoch}: Fashion Acc = {acc_b:.2f}%")

    # --- Final Evaluation ---
    print("\n[Phase 3] Final Evaluation (Catastrophic Forgetting Test)")

    print("Testing on Task A (MNIST)... (Should be preserved)")
    final_acc_a = trainer.predict(loader_a_test)

    print("Testing on Task B (Fashion-MNIST)... (Should be learned)")
    final_acc_b = trainer.predict(loader_b_test)

    print("\n=== Result Summary ===")
    print(f"Task A (Old Memory): {final_acc_a:.2f}%")
    print(f"Task B (New Memory): {final_acc_b:.2f}%")

    if final_acc_a > 50.0:
        print(">> SUCCESS: Catastrophic Forgetting Mitigated! (Hippocampus worked)")
        print("   The model learned Fashion-MNIST without forgetting MNIST.")
    else:
        print(">> FAIL: Catastrophic Forgetting Occurred.")


if __name__ == "__main__":
    run_continual_learning_demo()
