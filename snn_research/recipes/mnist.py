# ファイルパス: snn_research/recipes/mnist.py
# 日本語タイトル: MNIST SNN学習レシピ (Mypy Fixed)
# 概要: MNISTデータセットでSpiking CNNを学習させるレシピ。型ヒントを修正。

import os
import time
import logging
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

# パッケージ内インポート
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.base import BaseModel

# 外部ライブラリ
try:
    from spikingjelly.activation_based import functional as SJ_F
except ImportError:
    # フォールバック等は環境依存だが、ここではエラーを出して終了させる想定
    pass

logger = logging.getLogger("Recipe_MNIST")

# --- モデル定義 ---

class MNIST_SpikingCNN(BaseModel):
    """
    MNIST (1x28x28) に特化したSpiking CNNモデル。
    """

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

        # Layer 1: Conv(1->32) -> BN -> LIF -> MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = create_neuron(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv(32->64) -> BN -> LIF -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = create_neuron(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Flatten -> Linear -> LIF
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.lif3 = create_neuron(128)

        # Layer 4: Output
        self.fc2 = nn.Linear(128, num_classes, bias=False)

        self._init_weights()

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
            # LIF3の発火率を代表値として使用
            avg_spikes = self.lif3.spikes.mean().detach()

        return logits, avg_spikes, torch.tensor(0.0)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_mnist_training(config_override: Optional[Dict[str, Any]] = None):
    """
    MNIST学習を実行する関数。CLIから呼び出される。
    """
    # --- Config ---
    # 型注釈を追加してmypyの推論を助ける
    CONFIG: Dict[str, Any] = {
        "seed": 42,
        "epochs": 10,
        "batch_size": 128,
        "lr": 1e-3,
        "time_steps": 4,
        "num_classes": 10,
        "dataset_path": "./data/mnist",
        "output_json": "workspace/results/best_mnist_metrics.json",  # 出力先
        "neuron_config": {
            "tau_mem": 2.0,
            "base_threshold": 1.0,
            "v_reset": 0.0,
            "adaptation_strength": 0.1,
            "target_spike_rate": 0.1
        }
    }

    # 設定の上書き
    if config_override:
        CONFIG.update(config_override)

    set_seed(CONFIG["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ Using CPU")

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    os.makedirs(str(CONFIG["dataset_path"]), exist_ok=True)
    # ダウンロード時のログ抑制などは省略
    train_dataset = datasets.MNIST(
        root=str(CONFIG["dataset_path"]), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root=str(CONFIG["dataset_path"]), train=False, download=True, transform=transform)

    pin_memory = True if device.type == 'cuda' else False
    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(
        train_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(
        test_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Model ---
    model = MNIST_SpikingCNN(
        num_classes=int(CONFIG["num_classes"]),
        time_steps=int(CONFIG["time_steps"]),
        neuron_config=CONFIG["neuron_config"]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=float(CONFIG["lr"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(CONFIG["epochs"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # --- Training Loop ---
    best_acc = 0.0

    os.makedirs("workspace/results", exist_ok=True)

    for epoch in range(int(CONFIG["epochs"])):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", unit="batch", leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, avg_spikes, _ = model(inputs, return_spikes=True)
                loss = criterion(logits, targets)
                if avg_spikes.item() > 0:
                    loss += 0.01 * torch.abs(avg_spikes - 0.1)

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            SJ_F.reset_net(model)

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "Acc": f"{100.*correct/total:.1f}%"})

        scheduler.step()

        # --- Validation with Metrics ---
        model.eval()
        test_correct = 0
        test_total = 0
        total_latency = 0.0
        total_batches = 0
        spike_rates = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Latency Measurement
                start_event = time.time()
                logits, batch_avg_spikes, _ = model(inputs, return_spikes=True)
                end_event = time.time()

                # バッチサイズで割って1サンプルあたりの時間を概算（厳密にはオーバーヘッド含む）
                batch_latency = (end_event - start_event) * 1000  # ms
                total_latency += batch_latency / inputs.size(0)  # per sample
                total_batches += 1

                spike_rates.append(batch_avg_spikes.item())
                SJ_F.reset_net(model)

                _, predicted = logits.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        avg_latency = total_latency / total_batches if total_batches > 0 else 0.0
        avg_spike_rate = sum(spike_rates) / \
            len(spike_rates) if spike_rates else 0.0

        logger.info(
            f"Epoch {epoch+1}: Test Acc={test_acc:.2f}%, Latency={avg_latency:.2f}ms, SpikeRate={avg_spike_rate:.1%}")

        # --- Save Best Result ---
        if test_acc > best_acc:
            best_acc = test_acc

            # Model Save
            torch.save(model.state_dict(),
                       "workspace/results/best_mnist_snn.pth")

            # Metrics JSON Save
            metrics_data = {
                "accuracy": best_acc / 100.0,
                "latency_ms": avg_latency,
                "avg_spike_rate": avg_spike_rate,
                "estimated_energy_joules": 2.0e-3 * avg_spike_rate * 0.2,  # Simple estimation logic
                "epoch": epoch + 1
            }

            with open(str(CONFIG["output_json"]), "w") as f:
                json.dump(metrics_data, f, indent=4)

            logger.info(f"💾 Best Model & Metrics Saved! Acc: {best_acc:.2f}%")

            if best_acc >= 96.89:
                logger.info("🎉 GOAL REACHED!")

    logger.info(f"✅ Finished. Best Acc: {best_acc:.2f}%")