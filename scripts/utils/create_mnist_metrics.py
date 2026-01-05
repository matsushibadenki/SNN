# scripts/create_mnist_metrics.py
# 日本語タイトル: MNIST メトリクス生成スクリプト (Post-Training)
# 概要: 既存の学習済みモデル(best_mnist_snn.pth)をロードし、
#       推論精度・レイテンシ・スパイク率を測定してJSONファイルに出力する。
#       (再学習なしでverify_performance.pyを実行可能にするため)

import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# プロジェクトルートの設定
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 必要なモジュールのインポート
try:
    from spikingjelly.activation_based import functional as SJ_F
    from snn_research.core.neurons import AdaptiveLIFNeuron
    from snn_research.core.base import BaseModel
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Metrics_Gen")

# --- モデル定義 (学習時と同じ構造) ---


class MNIST_SpikingCNN(BaseModel):
    def __init__(self, num_classes=10, time_steps=4, neuron_config=None):
        super().__init__()
        self.time_steps = time_steps

        if neuron_config is None:
            neuron_config = {'features': 0,
                             'tau_mem': 2.0, 'base_threshold': 1.0}

        def create_neuron(features):
            params = neuron_config.copy()
            params['features'] = features
            valid_keys = [
                "features", "tau_mem", "base_threshold", "adaptation_strength",
                "target_spike_rate", "noise_intensity", "threshold_decay",
                "threshold_step", "v_reset", "homeostasis_rate", "refractory_period"
            ]
            clean_params = {k: v for k, v in params.items() if k in valid_keys}
            return AdaptiveLIFNeuron(**clean_params)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = create_neuron(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = create_neuron(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.lif3 = create_neuron(128)

        self.fc2 = nn.Linear(128, num_classes, bias=False)

    def forward(self, x, return_spikes=False):

        x_seq = x.unsqueeze(1).repeat(1, self.time_steps, 1, 1, 1)
        SJ_F.reset_net(self)

        outputs = []
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(True)

        for t in range(self.time_steps):
            x_t = x_seq[:, t]
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t, _ = self.lif1(x_t)
            x_t = self.pool1(x_t)

            x_t = self.conv2(x_t)
            x_t = self.bn2(x_t)
            x_t, _ = self.lif2(x_t)
            x_t = self.pool2(x_t)

            x_t = self.flatten(x_t)
            x_t = self.fc1(x_t)
            x_t, _ = self.lif3(x_t)
            x_t = self.fc2(x_t)
            outputs.append(x_t)

        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(False)

        outputs = torch.stack(outputs).permute(1, 0, 2)
        logits = outputs.mean(dim=1)

        avg_spikes = torch.tensor(0.0)
        if return_spikes:
            avg_spikes = self.lif3.spikes.mean().detach()

        return logits, avg_spikes, torch.tensor(0.0)


def main():
    model_path = "workspace/results/best_mnist_snn.pth"
    output_json = "workspace/results/best_mnist_metrics.json"

    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("Please run the training script first.")
        return

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"🚀 Using device: {device}")

    # データセット準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=0)

    # モデルロード
    logger.info("🏗️ Loading Model...")
    # 学習時と同じ設定
    neuron_config = {
        "tau_mem": 2.0, "base_threshold": 1.0, "v_reset": 0.0,
        "adaptation_strength": 0.1, "target_spike_rate": 0.1
    }
    model = MNIST_SpikingCNN(num_classes=10, time_steps=4,
                             neuron_config=neuron_config).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info("✅ Weights loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load weights: {e}")
        return

    # 評価実行
    model.eval()
    correct = 0
    total = 0
    total_latency = 0.0
    total_batches = 0
    spike_rates = []

    logger.info("📊 Starting Evaluation...")

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            # 計測
            start = time.time()
            logits, batch_avg_spikes, _ = model(inputs, return_spikes=True)
            end = time.time()

            batch_latency = (end - start) * 1000  # ms
            total_latency += batch_latency / inputs.size(0)
            total_batches += 1
            spike_rates.append(batch_avg_spikes.item())

            SJ_F.reset_net(model)

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_latency = total_latency / total_batches if total_batches > 0 else 0.0
    avg_spike_rate = sum(spike_rates) / \
        len(spike_rates) if spike_rates else 0.0

    logger.info(
        f"🏆 Result: Acc={acc:.2f}%, Latency={avg_latency:.2f}ms, SpikeRate={avg_spike_rate:.1%}")

    # JSON保存
    metrics_data = {
        "accuracy": acc / 100.0,
        "latency_ms": avg_latency,
        "avg_spike_rate": avg_spike_rate,
        "estimated_energy_joules": 2.0e-3 * avg_spike_rate * 0.2,
        "source_model": model_path
    }

    with open(output_json, "w") as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(f"💾 Metrics saved to: {output_json}")
    logger.info(
        "➡️ Now you can run: python scripts/verify_performance.py --metrics_json results/best_mnist_metrics.json")


if __name__ == "__main__":
    main()
