# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Fix: 再現性・堅牢性評価版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def set_seed(seed: int = 42):
    """再現性のためにSeedを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to: {seed}")

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10, prototypes=None, noise_level: float = 0.1):
    """
    パターン認識タスクデータ生成
    Args:
        noise_level: ノイズの強度 (ビット反転確率)
    """
    if prototypes is None:
        # 密度50%のランダムパターンを「正解」として定義
        prototypes = (torch.randn(out_features, in_features) > 0.0).float()
    
    x_data = []
    y_data = []
    
    for _ in range(num_samples):
        label = torch.randint(0, out_features, (1,)).item()
        pattern = prototypes[label].clone()
        
        # ノイズ注入
        noise = (torch.rand(in_features) < noise_level).float()
        noisy_pattern = torch.abs(pattern - noise)
        
        x_data.append(noisy_pattern)
        y_data.append(label)
    
    x = torch.stack(x_data)
    y = torch.tensor(y_data, dtype=torch.long)
    
    return x, y, prototypes

def run_simulation():
    set_seed(42)  # 再現性確保
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # パラメータ設定
    IN_FEATURES = 784
    HIDDEN_FEATURES = 4096 
    OUT_FEATURES = 10
    BATCH_SIZE = 128
    TOTAL_SAMPLES = 20000
    EPOCHS = 10
    TRAIN_NOISE = 0.1

    # モデル構築
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons (Liquid State Machine).")
    
    # --- 学習データ生成 ---
    print(f"Generating Training Data (Noise Level: {TRAIN_NOISE})...")
    x_train, y_train, shared_prototypes = generate_synthetic_data(
        num_samples=TOTAL_SAMPLES, 
        in_features=IN_FEATURES, 
        out_features=OUT_FEATURES,
        prototypes=None,
        noise_level=TRAIN_NOISE
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nStarting Readout Training Phase...")
    
    moving_avg_acc = 0.1
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        total_seen = 0
        
        core.train() # 学習モード
        
        for i, (data, target) in enumerate(loader):
            # 自律学習
            metrics = core.autonomous_step(data, target)
            
            # 評価用 (重み更新後)
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            moving_avg_acc = moving_avg_acc * 0.9 + batch_acc * 0.1
            
            if i % 50 == 0:
                w_out = core.output_gate.get_effective_weights()
                w_mean = w_out.abs().mean().item()
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:2d} [{i*BATCH_SIZE:5d}/{TOTAL_SAMPLES}] "
                      f"Acc: {batch_acc*100:4.1f}% (MA: {moving_avg_acc*100:4.1f}%) | "
                      f"W_Mean: {w_mean:.4f} | "
                      f"Time: {elapsed:.0f}s")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 99.5:
            print(">>> Target Accuracy Reached. Optimization Complete.")
            break
    
    print("\n=== Running Robustness Evaluation (Stress Test) ===")
    core.eval()
    
    # 複数のノイズレベルでテストを行い、堅牢性を確認する
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    TEST_SAMPLES = 2000
    
    print(f"{'Noise Level':<15} | {'Accuracy':<10} | {'Status':<10}")
    print("-" * 40)
    
    for noise in noise_levels:
        x_test, y_test, _ = generate_synthetic_data(
            num_samples=TEST_SAMPLES, 
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            prototypes=shared_prototypes,
            noise_level=noise
        )
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        test_correct = 0
        with torch.no_grad():
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            
            for data, target in test_loader:
                out = core(data)
                pred = out.argmax(dim=1)
                test_correct += (pred == target).float().sum().item()
                    
        final_acc = test_correct / TEST_SAMPLES * 100
        status = "Robust" if final_acc > 80.0 else "Weak"
        if noise == TRAIN_NOISE and final_acc > 95.0:
            status = "Excellent"
            
        print(f"{noise:<15.1f} | {final_acc:6.2f}%   | {status}")
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()
