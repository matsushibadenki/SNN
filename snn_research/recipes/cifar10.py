# ファイルパス: snn_research/recipes/cifar10.py
# 日本語タイトル: CIFAR-10 SNN High-Performance Recipe
# 概要: CIFAR-10で96%以上の精度を目指すためのSOTA学習レシピ（Mixup, Cutout, AutoAugment, SGD採用）。

import os
import time
import logging
import random
import numpy as np
import math

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

# AutoAugmentのインポート試行（torchvision >= 0.10）
try:
    from torchvision.transforms import AutoAugment, AutoAugmentPolicy
    HAS_AUTO_AUGMENT = True
except ImportError:
    HAS_AUTO_AUGMENT = False

logger = logging.getLogger("Recipe_CIFAR10")


# --- Utils: Cutout & Mixup ---

class Cutout(object):
    """Randomly mask out one or more patches from an image."""

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- Model Definitions ---

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

        # Add (Residual Connection) - SEW: ADD then FIRE
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
    CIFAR-10に最適化されたSEW ResNet-18。
    """

    def __init__(self, num_classes=10, time_steps=6, neuron_config=None):
        super().__init__()
        self.time_steps = time_steps

        if neuron_config is None:
            neuron_config = {'type': 'lif',
                             'tau_mem': 2.0, 'base_threshold': 1.0}

        # ニューロンパラメータの準備
        neuron_params = neuron_config.copy()

        # AdaptiveLIFNeuron用パラメータフィルタリング
        valid_keys = [
            "features", "tau_mem", "base_threshold", "adaptation_strength",
            "target_spike_rate", "noise_intensity", "threshold_decay",
            "threshold_step", "v_reset", "homeostasis_rate", "refractory_period"
        ]
        keys_to_remove = [
            k for k in neuron_params.keys() if k not in valid_keys]
        for k in keys_to_remove:
            neuron_params.pop(k, None)

        neuron_class = AdaptiveLIFNeuron

        self.in_channels = 64

        # Stem: 3x3 Conv, Stride 1 (CIFAR Optimized)
        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.in_channels)
        self.lif1 = neuron_class(features=self.in_channels, **neuron_params)

        # ResNet18 Structure
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
        # Time Repetition
        x_spikes = input_images.unsqueeze(
            1).repeat(1, self.time_steps, 1, 1, 1)

        SJ_F.reset_net(self)

        # Stem
        outputs = []
        if hasattr(self.lif1, 'set_stateful'):
            self.lif1.set_stateful(True)  # type: ignore

        for t in range(self.time_steps):
            x_t = x_spikes[:, t]
            out = self.conv1(x_t)
            out = self.norm1(out)
            out, _ = self.lif1(out)
            outputs.append(out)

        x_spikes = torch.stack(outputs, dim=1)

        # Layers
        x_spikes = self.layer1(x_spikes)
        x_spikes = self.layer2(x_spikes)
        x_spikes = self.layer3(x_spikes)
        x_spikes = self.layer4(x_spikes)

        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = x_spikes.detach().mean().item()

        # Classification (Rate Coding)
        x_mean = x_spikes.mean(dim=1)
        x_pooled = self.avgpool(x_mean)
        x_flat = torch.flatten(x_pooled, 1)
        logits = self.fc(x_flat)

        return logits, torch.tensor(avg_spikes_val).to(input_images.device), torch.tensor(0.0)


# --- Main Execution ---

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True  # 高速化のためTrue推奨


def run_cifar10_training(epochs=300, batch_size=128):
    """
    High-Performance CIFAR-10 Training Recipe.
    Aiming for >96% accuracy with SGD, Mixup, Cutout, AutoAugment.
    """
    # --- Advanced Config ---
    CONFIG = {
        "seed": 42,
        "epochs": 300,           # SNN SOTA Standard
        "batch_size": 128,       # SNN SOTA Standard
        "lr": 0.1,               # SGD Standard
        "min_lr": 1e-6,
        "momentum": 0.9,
        "weight_decay": 5e-4,    # ResNet Standard
        "time_steps": 6,         # 4 -> 6 for higher accuracy
        "num_classes": 10,
        "dataset_path": "./data/cifar10",
        "mixup_alpha": 1.0,      # Mixup有効
        "label_smoothing": 0.1,
        "cutout_holes": 1,
        "cutout_length": 16,
        "neuron_config": {
            "type": "lif",
            "tau_mem": 2.0,
            "base_threshold": 1.0,
            "v_reset": 0.0,
        }
    }

    # 引数で上書き（CLIからの指定を尊重しつつ、デフォルトを強力に）
    if epochs > 50:
        CONFIG["epochs"] = epochs
    if batch_size > 64:
        CONFIG["batch_size"] = batch_size

    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")
    if device.type == 'cpu':
        logger.warning(
            "⚠️ CPU training will be extremely slow for 300 epochs.")

    # --- Dataset with Strong Augmentation ---
    logger.info("📂 Preparing CIFAR-10 with Strong Augmentation...")

    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if HAS_AUTO_AUGMENT:
        logger.info("   -> Using AutoAugment (CIFAR10Policy)")
        transform_list.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
    else:
        logger.info("   -> AutoAugment not found. Skipping.")

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=CONFIG["cutout_holes"], length=CONFIG["cutout_length"])
    ])

    train_transform = transforms.Compose(transform_list)

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

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True)

    # --- Model ---
    logger.info(
        f"🏗️ Building CIFAR-10 SEW-ResNet18 (T={CONFIG['time_steps']})...")
    model = CIFAR10_SEWResNet(
        num_classes=CONFIG["num_classes"],
        time_steps=CONFIG["time_steps"],
        neuron_config=CONFIG["neuron_config"]
    ).to(device)

    # --- Optimizer & Loss ---
    # AdamWよりもSGDの方がCIFAR-10の最高精度が出やすい
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["lr"],
                          momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
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
                # Mixup Execution
                mixed_inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, CONFIG["mixup_alpha"], device)

                logits, avg_spikes, _ = model(mixed_inputs, return_spikes=True)

                loss = mixup_criterion(
                    criterion, logits, targets_a, targets_b, lam)

                # Reg loss (Optional: keep low)
                reg_loss = 0.0
                if avg_spikes.item() > 0:
                    reg_loss = 1e-5 * torch.abs(avg_spikes - 0.1)  # 弱い正則化

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
            # Mixup時の精度計算は概算になるが、ここでは通常のターゲットとの一致を見る（参考値）
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "LR": f"{optimizer.param_groups[0]['lr']:.4f}"})

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
            f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Test Acc={test_acc:.2f}% (Best: {max(best_acc, test_acc):.2f}%)")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("workspace/results", exist_ok=True)
            torch.save(model.state_dict(),
                       "workspace/results/best_cifar10_snn.pth")
            if best_acc >= 96.0:
                logger.info("🏆 Target Reached: Accuracy >= 96.0%")

    total_time = (time.time() - start_time) / 60 / 60
    logger.info(
        f"✅ Finished. Total Time: {total_time:.2f} hours. Best Acc: {best_acc:.2f}%")
