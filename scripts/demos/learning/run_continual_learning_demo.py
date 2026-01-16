# ファイルパス: scripts/demos/learning/run_continual_learning_demo.py
# Title: Continual Learning Demo (Final Stable Version)
# Description:
#   MNISTとFashion-MNISTを連続学習するデモ。
#   [Fix] 終了時の Segmentation Fault を防ぐため、明示的なメモリ解放処理(Cleanup)を追加。
#   [Fix] 動作安定性のため環境変数を設定。

import sys
import os
import gc

# 動作安定化のための環境変数設定
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # マルチスレッド競合によるSegfault防止

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from collections import deque
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../')))

from snn_research.core.ensemble_scal import EnsembleSCAL
from snn_research.training.trainers.forward_forward import ForwardForwardTrainer, SpikingForwardForwardLayer, ForwardForwardLayer


# --- 共通コンポーネント ---

class SCALFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features, n_models=5, device='cpu'):
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
        print(f"  SCAL training ({epochs} epochs)...")
        for _ in range(epochs):
            for data, _ in data_loader:
                if not data.is_contiguous():
                    data = data.contiguous()
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)


class HippocampalReplayBuffer:
    """
    長期記憶バッファ (Reservoir Sampling版)。
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_count = 0

    def push(self, features, label):
        features_cpu = features.detach().cpu()
        label_cpu = label.detach().cpu()
        
        batch_size = features.size(0)
        
        for i in range(batch_size):
            item = (features_cpu[i].clone(), label_cpu[i].clone())
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                m = random.randint(0, self.seen_count)
                if m < self.capacity:
                    self.buffer[m] = item
            
            self.seen_count += 1

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None
        
        batch = random.sample(self.buffer, batch_size)
        features, labels = zip(*batch)
        return torch.stack(features), torch.stack(labels)

    def clear(self):
        """メモリ解放用"""
        self.buffer.clear()

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

        features = features.to(self.device)

        if self.model.training:
            self.replay_buffer.push(features, y)

        x_pos = self.overlay_y_on_x(features, y)
        y_fake = (y + torch.randint(1, self.num_classes,
                  (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        return x_pos, x_neg

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        if epoch:
            self.current_epoch = epoch
        total_loss = 0.0

        self.model.train()

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data, target = data.to(self.device), target.to(self.device)

            x_pos_curr, x_neg_curr = self.generate_negative_data(data, target)

            # Interleaved Replay
            batch_size = data.size(0)
            replay_sample = self.replay_buffer.sample(batch_size)
            
            x_pos_final = x_pos_curr
            x_neg_final = x_neg_curr

            if replay_sample[0] is not None:
                features_mem, labels_mem = replay_sample
                features_mem = features_mem.to(self.device)
                labels_mem = labels_mem.to(self.device)

                x_pos_mem = self.overlay_y_on_x(features_mem, labels_mem)
                y_fake_mem = (labels_mem + torch.randint(1, self.num_classes,
                          (len(labels_mem),)).to(self.device)) % self.num_classes
                x_neg_mem = self.overlay_y_on_x(features_mem, y_fake_mem)

                x_pos_final = torch.cat([x_pos_curr, x_pos_mem], dim=0)
                x_neg_final = torch.cat([x_neg_curr, x_neg_mem], dim=0)

            batch_loss = 0.0

            for layer in self.execution_pipeline:
                if isinstance(layer, (ForwardForwardLayer, SpikingForwardForwardLayer)):
                    layer_loss, _, _ = layer.train_step(x_pos_final, x_neg_final)
                    batch_loss += layer_loss

                    with torch.no_grad():
                        x_pos_final = layer(x_pos_final).detach()
                        x_neg_final = layer(x_neg_final).detach()
                else:
                    x_pos_final = layer(x_pos_final)
                    x_neg_final = layer(x_neg_final)

            total_loss += batch_loss

        return {"train_loss": total_loss / len(train_loader)}

    def train_sleep_phase_with_replay(self, batch_size=64):
        for layer in self.execution_pipeline:
            layer.train()

        sampled_data = self.replay_buffer.sample(batch_size)
        if sampled_data is None or sampled_data[0] is None:
            return 0.0
        
        features_mem, labels_mem = sampled_data
        features_mem = features_mem.to(self.device)
        labels_mem = labels_mem.to(self.device)

        x_p = self.overlay_y_on_x(features_mem, labels_mem)
        y_fake = (labels_mem + torch.randint(1, self.num_classes,
                  (len(labels_mem),)).to(self.device)) % self.num_classes
        x_n = self.overlay_y_on_x(features_mem, y_fake)

        total_loss = 0.0
        for layer in self.execution_pipeline:
            if isinstance(layer, (SpikingForwardForwardLayer, ForwardForwardLayer)):
                l_loss, _, _ = layer.train_step(x_p, x_n)
                total_loss += l_loss
                with torch.no_grad():
                    x_p = layer.forward(x_p).detach()
                    x_n = layer.forward(x_n).detach()
            else:
                x_p = layer(x_p)
                x_n = layer(x_n)
        return total_loss

    def predict(self, data_loader: DataLoader) -> float:
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

                features = self.feature_extractor(data)
                features = features.to(self.device)

                class_goodness = torch.zeros(
                    data.size(0), self.num_classes, device=self.device)

                for label_idx in range(self.num_classes):
                    temp_labels = torch.full(
                        (data.size(0),), label_idx, dtype=torch.long, device=self.device)
                    h = self.overlay_y_on_x(features, temp_labels)

                    for layer in self.execution_pipeline:
                        if isinstance(layer, (SpikingForwardForwardLayer, ForwardForwardLayer)):
                            h = layer.forward(h)
                            if isinstance(layer, SpikingForwardForwardLayer):
                                g = h.mean(dim=1).pow(2).mean(dim=1)
                            else:
                                dims = list(range(1, h.dim()))
                                g = h.pow(2).mean(dim=dims)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)

                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100.0 * correct / total


def run_continual_learning_demo():
    print("=== Continual Learning Experiment: MNIST -> Fashion-MNIST ===")

    device = "cpu"
    print(f"Using device: {device} (Forced for stability)")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))
    os.makedirs(data_dir, exist_ok=True)

    print("Loading Task A: MNIST (Digits)...")
    mnist_train = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, transform=transform)
    loader_a_train = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=0)
    loader_a_test = DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=0)

    print("Loading Task B: Fashion-MNIST (Clothing)...")
    fashion_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(
        data_dir, train=False, transform=transform)
    loader_b_train = DataLoader(fashion_train, batch_size=64, shuffle=True, num_workers=0)
    loader_b_test = DataLoader(fashion_test, batch_size=100, shuffle=False, num_workers=0)

    scal_dim = 1000
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
    scal.fit(loader_a_train, epochs=3)

    for epoch in range(1, 4):
        print(f"Training Epoch {epoch}...")
        trainer.train_epoch(loader_a_train, epoch)
        
        print("  Sleeping (Consolidating Memory)...")
        for _ in range(10):
            trainer.train_sleep_phase_with_replay()

        if epoch % 1 == 0:
            print("  Predicting Task A...")
            acc_a = trainer.predict(loader_a_test)
            print(f"  Epoch {epoch}: MNIST Acc = {acc_a:.2f}%")

    print("Task A Learning Finished. Retaining memory buffer...")

    # --- Task B: Fashion-MNIST Learning ---
    print("\n[Phase 2] Learning Task B: Fashion-MNIST (while remembering MNIST)")
    print("  (Keeping Feature Extractor Frozen to prevent Perceptual Forgetting)")

    for epoch in range(1, 4):
        print(f"Training Epoch {epoch}...")
        trainer.train_epoch(loader_b_train, epoch)

        print("  Sleeping (Consolidating Mixed Memory)...")
        for _ in range(10):
            trainer.train_sleep_phase_with_replay()

        if epoch % 1 == 0:
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

    # [Fix] 明示的なクリーンアップ (Segfault回避)
    print("\nCleaning up resources...")
    trainer.replay_buffer.clear()
    del trainer
    del scal
    del snn_model
    gc.collect()
    print("Cleanup complete.")


if __name__ == "__main__":
    run_continual_learning_demo()