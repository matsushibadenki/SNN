# ファイルパス: snn_research/recipes/cifar10.py
# 日本語タイトル: CIFAR-10 SNN学習レシピ
# 概要: CIFAR-10に最適化したSEW-ResNetを学習させるレシピ。snn-cliから呼び出される。

import os
import sys
import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.base import BaseModel

try:
    from spikingjelly.activation_based import functional as SJ_F
except ImportError:
    pass

logger = logging.getLogger("Recipe_CIFAR10")

# --- 1. ローカル修正版モデル定義 (CIFAR-10 Optimized) ---

class SEWResidualBlock(nn.Module):
    """SEW (Spike-Element-Wise) 残差ブロック"""

    def __init__(self, in_channels, out_channels, stride, neuron_class, neuron_params):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.lif1 = neuron_class(features=out_channels, **neuron_params)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.lif_shortcut = neuron_class(
            features=out_channels, **neuron_params)
        self.lif_out = neuron_class(features=out_channels, **neuron_params)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        # x_spike: [B, T, C, H, W]
        B, T, C_in, H, W = x_spike.shape

        # --- Shortcut Path ---
        x_flat = x_spike.flatten(0, 1)  # (B*T, C, H, W)

        if self.downsample:
            identity = self.downsample(x_flat)
        else:
            identity = x_flat

        # Shortcut path neurons
        self.forward_neuron_time_series(self.lif_shortcut, identity, B, T)

        # --- Main Path ---
        out = self.conv1(x_flat)
        out = self.norm1(out)
        out = self.forward_neuron_time_series(self.lif1, out, B, T)  # LIF1

        out_flat = out.flatten(0, 1)
        out = self.conv2(out_flat)
        out = self.norm2(out)

        # Add (Residual Connection)
        out = out + identity

        # Final LIF
        return self.forward_neuron_time_series(self.lif_out, out, B, T)

    def forward_neuron_time_series(self, neuron_module, x_flat, B, T):
        _, C, H, W = x_flat.shape
        x_time = x_flat.view(B, T, C, H, W)

        outputs = []
        if hasattr(neuron_module, 'set_stateful'):
            neuron_module.set_stateful(True)  # type: ignore

        for t in range(T):
            out = neuron_module(x_time[:, t])
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)


class CIFAR10_SEWResNet(BaseModel):
    """
    CIFAR-10に最適化されたSEW ResNet。
    ImageNet用のStem（7x7 Conv, Stride 2）を廃止し、3x3 Conv, Stride 1を採用。
    """

    def __init__(self, num_classes=10, time_steps=8, neuron_config=None):
        super().__init__()
        self.time_steps = time_steps

        if neuron_config is None:
            neuron_config = {'type': 'lif',
                             'tau_mem': 2.0, 'base_threshold': 1.0}

        # ニューロンパラメータの準備
        neuron_params = neuron_config.copy()

        # AdaptiveLIFNeuronの__init__引数として無効なものを削除
        valid_keys = [
            "features", "tau_mem", "base_threshold", "adaptation_strength",
            "target_spike_rate", "noise_intensity", "threshold_decay",
            "threshold_step", "v_reset", "homeostasis_rate", "refractory_period"
        ]
        # 不要なキーを削除 (type, detach_reset 等)
        keys_to_remove = [
            k for k in neuron_params.keys() if k not in valid_keys]
        for k in keys_to_remove:
            neuron_params.pop(k, None)

        neuron_class = AdaptiveLIFNeuron  # プロジェクト内のAdaptiveLIFを使用

        self.in_channels = 64

        # --- CIFAR-10 Optimization: 3x3 Conv, Stride 1, No MaxPool ---
        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.in_channels)
        self.lif1 = neuron_class(features=self.in_channels, **neuron_params)

        # ResNet18構造 (2-2-2-2 blocks)
        self.layer1 = self._make_layer(
            self.in_channels, 64, 2, 1, neuron_class, neuron_params)
        self.layer2 = self._make_layer(
            64, 128, 2, 2, neuron_class, neuron_params)
        self.layer3 = self._make_layer(
            128, 256, 2, 2, neuron_class, neuron_params)
        self.layer4 = self._make_layer(
            256, 512, 2, 2, neuron_class, neuron_params)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, neuron_class, neuron_params):
        layers = []
        layers.append(SEWResidualBlock(in_channels, out_channels,
                      stride, neuron_class, neuron_params))
        for _ in range(1, num_blocks):
            layers.append(SEWResidualBlock(
                out_channels, out_channels, 1, neuron_class, neuron_params))
        return nn.Sequential(*layers)

    def forward(self, input_images, return_spikes=False, **kwargs):
        # input_images: (B, C, H, W)
        B, C, H, W = input_images.shape

        # 入力を時間軸方向にリピート: (B, T, C, H, W)
        x_spikes = input_images.unsqueeze(
            1).repeat(1, self.time_steps, 1, 1, 1)

        # ニューロン状態のリセット
        SJ_F.reset_net(self)

        # Stem処理
        outputs = []
        if hasattr(self.lif1, 'set_stateful'):
            self.lif1.set_stateful(True)  # type: ignore

        for t in range(self.time_steps):
            x_t = x_spikes[:, t]  # (B, C, H, W)
            out = self.conv1(x_t)
            out = self.norm1(out)
            out, _ = self.lif1(out)  # (B, C, H, W)
            outputs.append(out)

        x_spikes = torch.stack(outputs, dim=1)  # (B, T, 64, 32, 32)

        # Residual Layers
        x_spikes = self.layer1(x_spikes)
        x_spikes = self.layer2(x_spikes)
        x_spikes = self.layer3(x_spikes)
        x_spikes = self.layer4(x_spikes)

        # スパイク数集計
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = x_spikes.detach().mean().item()

        # Classification (Rate Coding: Average over time)
        x_mean = x_spikes.mean(dim=1)  # (B, 512, H', W')
        x_pooled = self.avgpool(x_mean)  # (B, 512, 1, 1)
        x_flat = torch.flatten(x_pooled, 1)  # (B, 512)
        logits = self.fc(x_flat)

        return logits, torch.tensor(avg_spikes_val).to(input_images.device), torch.tensor(0.0)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_cifar10_training(epochs=50, batch_size=64):
    """
    CIFAR-10学習を実行する関数。CLIから呼び出される。
    """
    # --- Config ---
    CONFIG = {
        "seed": 42,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 1e-4,
        "time_steps": 4,
        "num_classes": 10,
        "dataset_path": "./data/cifar10",
        "neuron_config": {
            "type": "lif",
            "tau_mem": 2.0,
            "base_threshold": 1.0,
            "v_reset": 0.0,
            # "detach_reset": True # AdaptiveLIFNeuronでは非対応（デフォルトでdetachされる）ため削除
        }
    }

    set_seed(CONFIG["seed"])

    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ Using CPU (Slow)")

    # --- Dataset ---
    logger.info("📂 Preparing CIFAR-10 Dataset...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    os.makedirs(CONFIG["dataset_path"], exist_ok=True)
    train_dataset = datasets.CIFAR10(
        root=CONFIG["dataset_path"], train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(
        root=CONFIG["dataset_path"], train=False, download=True, transform=test_transform)

    pin_memory = True if device.type == 'cuda' else False
    num_workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Model ---
    logger.info(
        f"🏗️ Building Optimized CIFAR-10 SEW-ResNet (T={CONFIG['time_steps']})...")
    model = CIFAR10_SEWResNet(
        num_classes=CONFIG["num_classes"],
        time_steps=CONFIG["time_steps"],
        neuron_config=CONFIG["neuron_config"]
    ).to(device)

    logger.info(
        f"   -> Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # --- Training Loop ---
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
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

                reg_loss = 0.0
                if avg_spikes.item() > 0:
                    reg_loss = 0.0001 * torch.abs(avg_spikes - 0.05)

                total_loss = loss + reg_loss

            if device.type == 'cuda':
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            SJ_F.reset_net(model)

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "Acc": f"{100.*correct/total:.1f}%"})

        scheduler.step()

        # --- Validation ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _, _ = model(inputs)
                SJ_F.reset_net(model)
                _, predicted = logits.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        logger.info(
            f"Epoch {epoch+1}: Train Acc={100.*correct/total:.2f}%, Test Acc={test_acc:.2f}% (Best: {max(best_acc, test_acc):.2f}%)")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("workspace/results", exist_ok=True)
            torch.save(model.state_dict(),
                       "workspace/results/best_cifar10_snn.pth")
            if best_acc > 58.2:
                logger.info("🚀 Goal Reached: Accuracy > 58.2%")

    total_time = (time.time() - start_time) / 60
    logger.info(
        f"✅ Finished. Total Time: {total_time:.1f} min. Best Acc: {best_acc:.2f}%")