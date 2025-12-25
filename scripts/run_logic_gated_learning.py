# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: Stable & Robust)
# 内容: カリキュラム学習、LR減衰、ロバスト性評価を含む完全な学習ループ

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np
from typing import Union, Tuple

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

def generate_synthetic_data(num_samples: int = 5000, 
                          in_features: int = 784, 
                          out_features: int = 10, 
                          prototypes=None, 
                          noise_level: Union[float, Tuple[float, float]] = 0.1):
    """
    合成データの生成 (XOR Noise)
    """
    if prototypes is None:
        prototypes = (torch.randn(out_features, in_features) > 0.0).float()
    
    x_data = []
    y_data = []
    
    for _ in range(num_samples):
        label = torch.randint(0, out_features, (1,)).item()
        pattern = prototypes[label].clone()
        
        if isinstance(noise_level, (tuple, list)):
            current_noise = random.uniform(noise_level[0], noise_level[1])
        else:
            current_noise = noise_level
            
        noise = (torch.rand(in_features) < current_noise).float()
        noisy_pattern = torch.abs(pattern - noise)
        
        x_data.append(noisy_pattern)
        y_data.append(label)
    
    x = torch.stack(x_data)
    y = torch.tensor(y_data, dtype=torch.long)
    
    return x, y, prototypes

def run_simulation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    IN_FEATURES = 784
    HIDDEN_FEATURES = 10000 
    OUT_FEATURES = 10
    BATCH_SIZE = 128
    TOTAL_SAMPLES = 20000
    EPOCHS = 40
    
    INITIAL_LR = 0.03
    
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons.")
    print(f"Training Logic: Adaptive Threshold, Hard Top-K, Stable LR.")
    
    _, _, shared_prototypes = generate_synthetic_data(num_samples=1, in_features=IN_FEATURES, out_features=OUT_FEATURES)
    shared_prototypes = shared_prototypes.to(device)

    print("\nStarting Curriculum Training Phase...")
    print(f"{'Epoch':<6} | {'Noise Range':<15} | {'Acc':<6} | {'Loss':<6} | {'LR':<7} | {'R_Spk%':<6} | {'V_Mean':<6} | {'Time'}")
    print("-" * 95)
    
    start_time = time.time()
    current_lr = INITIAL_LR
    
    for epoch in range(EPOCHS):
        # カリキュラム学習設定
        if epoch < 5:
            current_noise_range = (0.0, 0.20) 
        elif epoch < 15:
            current_noise_range = (0.0, 0.40) 
        elif epoch < 25:
            current_noise_range = (0.0, 0.45)
        else:
            current_noise_range = (0.0, 0.49)
            
        # 学習率の減衰 (0.97倍/epoch)
        current_lr = INITIAL_LR * (0.97 ** epoch)
            
        x_train, y_train, _ = generate_synthetic_data(
            num_samples=TOTAL_SAMPLES, 
            in_features=IN_FEATURES, 
            out_features=OUT_FEATURES,
            prototypes=shared_prototypes,
            noise_level=current_noise_range
        )
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        epoch_correct = 0
        total_seen = 0
        epoch_loss = 0.0
        
        core.train()
        
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target, learning_rate=current_lr)
            acc = metrics["accuracy"]
            epoch_correct += acc * data.size(0)
            total_seen += data.size(0)
            epoch_loss += metrics["loss"]
            
        epoch_acc = epoch_correct / total_seen * 100
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        
        print(f"{epoch+1:<6} | {str(current_noise_range):<15} | "
              f"{epoch_acc:5.1f}% | "
              f"{avg_loss:6.4f} | "
              f"{current_lr:.5f} | "
              f"{metrics['res_density']*100:5.1f}% | "
              f"{metrics['out_v_mean']:6.3f} | "
              f"{elapsed:.0f}s")
        
    print("Optimization Complete.")
    
    print("\n=== Running Robustness Evaluation (Stress Test) ===")
    core.eval()
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    TEST_SAMPLES = 2000
    
    print(f"{'Noise':<6} | {'Acc':<7} | {'Loss':<6} | {'O_Spk%':<6} | {'V_Mean':<6} | {'Status'}")
    print("-" * 65)
    
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
        total_loss = 0.0
        total_spikes = 0.0
        total_v_mean = 0.0
        
        with torch.no_grad():
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            
            for data, target in test_loader:
                out = core(data)
                
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                loss = (target_onehot - out).pow(2).mean().item()
                total_loss += loss * data.size(0)
                
                total_spikes += out.mean().item() * data.size(0)
                total_v_mean += core.output_gate.membrane_potential.mean().item() * data.size(0)

                pred = out.argmax(dim=1)
                test_correct += (pred == target).float().sum().item()
                    
        final_acc = test_correct / TEST_SAMPLES * 100
        avg_loss = total_loss / TEST_SAMPLES
        avg_spk = (total_spikes / TEST_SAMPLES) * 100
        avg_v = total_v_mean / TEST_SAMPLES
        
        status = "Robust" if final_acc > 80.0 else "Weak"
        if final_acc > 98.0: status = "Excellent"
        if noise >= 0.5:
            if final_acc < 15.0: status = "Theoretical Limit (OK)"
            else: status = "Suspiciously High"

        print(f"{noise:<6.2f} | {final_acc:6.1f}% | {avg_loss:6.4f} | {avg_spk:5.1f}% | {avg_v:6.3f} | {status}")
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()
