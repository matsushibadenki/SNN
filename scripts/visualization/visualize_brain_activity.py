# ファイルパス: scripts/rvisualize_brain_activity.py
# Title: rvisualize_brain_activity
# Description:

from snn_research.core.ensemble_scal import EnsembleSCAL
from snn_research.training.trainers.forward_forward import SpikingForwardForwardLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# === モデル定義 (前回の実験と同じ構成) ===


class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=3, device='cpu'):
        super().__init__()
        self.scal = EnsembleSCAL(
            in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        self.norm = nn.LayerNorm(out_features).to(device)

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return self.norm(out['output'])

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print("initializing Visual Cortex (SCAL)...")
        for _ in range(epochs):
            for data, _ in data_loader:
                if not data.is_contiguous():
                    data = data.contiguous()
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)


class BrainVisualizer:
    def __init__(self, device):
        self.device = device
        self.feature_extractor = None
        self.snn_layers = []
        self.num_classes = 10

    def setup_model(self):
        print("Building the Brain...")
        # 1. 視覚野 (SCAL)
        self.feature_extractor = SCALFeatureExtractor(
            784, 1000, n_models=3, device=self.device)

        # 2. 大脳皮質 (SNN Layers)
        # 可視化しやすいよう、少し小さめのネットワークにする
        layer1 = nn.Sequential(nn.Linear(1000, 500))
        layer2 = nn.Sequential(nn.Linear(500, 500))

        self.snn_layers = [
            SpikingForwardForwardLayer(
                layer1, threshold=0.15, time_steps=30).to(self.device),
            SpikingForwardForwardLayer(
                layer2, threshold=0.15, time_steps=30).to(self.device)
        ]

    def overlay_y_on_x(self, x, y):
        x_mod = x.clone()
        y_one_hot = F.one_hot(
            y, num_classes=self.num_classes).float().to(self.device)
        scale_factor = 1.5
        y_signal = y_one_hot * scale_factor

        # [B, Features] or [B, T, Features]
        if x_mod.dim() == 2:
            x_mod[:, :self.num_classes] = y_signal
        elif x_mod.dim() == 3:
            y_sig_expanded = y_signal.unsqueeze(1)
            x_mod[:, :, :self.num_classes] += y_sig_expanded
        return x_mod

    def train_quick(self, loader):
        print("Quick Training to activate neurons (1 Epoch)...")
        # SCAL初期化
        self.feature_extractor.fit(loader, epochs=1)

        # SNN学習
        for layer in self.snn_layers:
            layer.train()

        from tqdm import tqdm
        pbar = tqdm(loader, total=len(loader))

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            # 特徴抽出
            features = self.feature_extractor(data)
            features = features.to(self.device)

            # Positive / Negative 生成
            x_pos = self.overlay_y_on_x(features, target)
            y_fake = (target + torch.randint(1, 10,
                      (len(target),)).to(self.device)) % 10
            x_neg = self.overlay_y_on_x(features, y_fake)

            loss_sum = 0
            # 層ごとに学習
            h_pos, h_neg = x_pos, x_neg
            for layer in self.snn_layers:
                loss, _, _ = layer.train_step(h_pos, h_neg)
                loss_sum += loss
                with torch.no_grad():
                    h_pos = layer.forward(h_pos).detach()
                    h_neg = layer.forward(h_neg).detach()

            pbar.set_description(f"Loss: {loss_sum:.4f}")

    def visualize(self, loader):
        print("\n=== Visualizing Brain Activity ===")
        # テストデータを1つだけ取り出す
        data, target = next(iter(loader))
        # 最初のサンプルを使用
        sample_idx = 0
        img = data[sample_idx]
        label = target[sample_idx].item()

        print(f"Input Stimulus: Digit '{label}'")

        data = data.to(self.device)
        target = target.to(self.device)

        # 1. 視覚野の活動 (特徴抽出)
        features = self.feature_extractor(data)  # [B, 1000]
        feature_vec = features[sample_idx].detach().cpu().numpy()

        # 2. 大脳皮質の活動 (スパイク列)
        # 正解ラベルを与えた時の反応を見る
        features = features.to(self.device)
        temp_label = torch.tensor([label], device=self.device)
        h = self.overlay_y_on_x(features[sample_idx:sample_idx+1], temp_label)

        spike_data = []

        for i, layer in enumerate(self.snn_layers):
            # forwardでスパイク [B, T, F] を取得
            with torch.no_grad():
                spikes = layer.forward(h)  # [1, 30, 500]
                h = spikes.detach()  # 次の層へ

            # [Time, Neurons] に変換
            spike_raster = spikes[0].cpu().numpy()
            spike_data.append(spike_raster)
            print(
                f"Layer {i+1} Spikes captured. Active neurons: {np.sum(spike_raster) / spike_raster.size * 100:.2f}%")

        # --- Plotting ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"Brain Activity for Input '{label}'", fontsize=16, color='white')

        # A. Input Image
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img.squeeze().cpu().numpy(), cmap='gray')
        ax1.set_title("Retina (Input)", color='cyan')
        ax1.axis('off')

        # B. Visual Cortex (SCAL Features)
        ax2 = fig.add_subplot(2, 2, 2)
        # 特徴ベクトルを 25x40 にリシェイプしてヒートマップ化
        sns.heatmap(feature_vec.reshape(25, 40),
                    cmap='viridis', ax=ax2, cbar=False)
        ax2.set_title("Visual Cortex (SCAL Features)", color='lime')
        ax2.axis('off')

        # C. Cortex Layer 1 Spikes (Raster Plot)
        ax3 = fig.add_subplot(2, 2, 3)
        raster1 = spike_data[0].T  # [Neurons, Time]
        # ニューロン数が多すぎるので、活発な上位100個だけ表示
        active_indices = np.argsort(raster1.sum(axis=1))[-100:]
        ax3.imshow(raster1[active_indices], aspect='auto',
                   cmap='binary', interpolation='nearest')
        ax3.set_title("Cortex Layer 1 (Spike Raster)", color='yellow')
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Active Neurons")

        # D. Cortex Layer 2 Spikes (Raster Plot)
        ax4 = fig.add_subplot(2, 2, 4)
        raster2 = spike_data[1].T
        active_indices2 = np.argsort(raster2.sum(axis=1))[-100:]
        ax4.imshow(raster2[active_indices2], aspect='auto',
                   cmap='magma', interpolation='nearest')
        ax4.set_title("Cortex Layer 2 (Deep Reasoning)", color='magenta')
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Active Neurons")

        save_path = "workspace/brain_activity.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\n>> Visualization saved to '{save_path}'")
        print(">> Open this image to see your AI's brain in action!")


def run_visualization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Visualize
    viz = BrainVisualizer(device)
    viz.setup_model()
    viz.train_quick(loader)  # 脳を目覚めさせる
    viz.visualize(loader)   # 覗く


if __name__ == "__main__":
    run_visualization()
