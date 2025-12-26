# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: Signal Boosting)
# 内容: Signal Boost機能と長期微調整により、Acc 88%の壁を突破する。

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
try:
    from snn_research.core.hybrid_core import HybridNeuromorphicCore
    from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
except ImportError:
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    print(f"Seed set to: {seed}")

def generate_synthetic_data(num_samples: int = 5000, 
                          in_features: int = 784, 
                          out_features: int = 10, 
                          prototypes=None, 
                          noise_level: Union[float, Tuple[float, float]] = 0.1):
    device = prototypes.device if prototypes is not None else torch.device('cpu')

    if prototypes is None:
        prototypes = (torch.randn(out_features, in_features, device=device) > 0.0).float()
    
    labels = torch.randint(0, out_features, (num_samples,), device=device)
    patterns = prototypes[labels]
    
    if isinstance(noise_level, (tuple, list)):
        probs = torch.rand(num_samples, 1, device=device) * (noise_level[1] - noise_level[0]) + noise_level[0]
    else:
        probs = torch.full((num_samples, 1), noise_level, device=device)
    
    noise_mask = (torch.rand(num_samples, in_features, device=device) < probs).float()
    x = torch.abs(patterns - noise_mask)
    y = labels
    
    return x, y, prototypes

def run_simulation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    IN_FEATURES = 784
    OUT_FEATURES = 10
    BATCH_SIZE = 4096 
    TOTAL_SAMPLES = 60000
    
    # [修正] カリキュラム: 最終段階のEpoch数を30に増やし、Signal Boostの効果を浸透させる
    curriculum_stages = [
        {'range': (0.0, 0.30), 'epochs': 10, 'lr': 0.1},
        {'range': (0.2, 0.40), 'epochs': 10, 'lr': 0.05},
        {'range': (0.35, 0.45), 'epochs': 15, 'lr': 0.02}, 
        {'range': (0.40, 0.46), 'epochs': 15, 'lr': 0.005}, 
        # 最終仕上げ: 30 Epochsかけてじっくりと発火タイミングを調整
        {'range': (0.43, 0.47), 'epochs': 30, 'lr': 0.002}, 
    ]
    
    layer = LogicGatedSNN(IN_FEATURES, OUT_FEATURES, mode='readout').to(device)
    
    print(f"\nModel initialized: LogicGatedSNN (Statistical Averaging Mode)")
    print(f"Training Logic: Granular Curriculum Learning (Signal Boosting).")
    
    _, _, shared_prototypes = generate_synthetic_data(num_samples=1, in_features=IN_FEATURES, out_features=OUT_FEATURES)
    shared_prototypes = shared_prototypes.to(device)

    print("\nStarting Curriculum Training Phase...")
    print(f"{'Epoch':<6} | {'Noise Range':<15} | {'Acc':<6} | {'Loss':<6} | {'LR':<8} | {'Temp':<6} | {'V_Mean':<6} | {'Time'}")
    print("-" * 96)
    
    start_time = time.time()
    
    current_epoch = 0
    
    # カリキュラムステージごとのループ
    for stage in curriculum_stages:
        noise_range = stage['range']
        stage_epochs = stage['epochs']
        lr = stage['lr']
        
        for _ in range(stage_epochs):
            current_epoch += 1
            
            x_train, y_train, _ = generate_synthetic_data(
                num_samples=TOTAL_SAMPLES, 
                in_features=IN_FEATURES, 
                out_features=OUT_FEATURES,
                prototypes=shared_prototypes,
                noise_level=noise_range
            )
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type=='cuda'))

            epoch_correct = 0
            total_seen = 0
            epoch_loss = 0.0
            
            layer.train()
            
            for data, target in loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                out = layer(data)
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                layer.update_plasticity(data, out, target_onehot, learning_rate=lr)
                
                loss = (target_onehot - out).pow(2).mean().item()
                pred = out.argmax(dim=1)
                acc = (pred == target).float().sum().item()
                
                epoch_correct += acc
                total_seen += data.size(0)
                epoch_loss += loss
                
            epoch_acc = epoch_correct / total_seen * 100
            avg_loss = epoch_loss / len(loader)
            elapsed = time.time() - start_time
            
            v_mean_val = layer.membrane_potential.mean().item()
            temp_val = layer.adaptive_threshold.mean().item()
            
            print(f"{current_epoch:<6} | {str(noise_range):<15} | "
                  f"{epoch_acc:5.1f}% | "
                  f"{avg_loss:6.4f} | "
                  f"{lr:.6f} | "
                  f"{temp_val:5.1f}  | "
                  f"{v_mean_val:6.3f} | "
                  f"{elapsed:.0f}s")
        
    print("Optimization Complete.")
    
    print("\n=== Running Robustness Evaluation (Stress Test) ===")
    layer.eval()
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5]
    TEST_SAMPLES = 20000 
    
    print(f"{'Noise':<6} | {'Acc':<7} | {'Loss':<6} | {'V_Mean':<6} | {'Status'}")
    print("-" * 55)
    
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
        total_v_mean = 0.0
        
        with torch.no_grad():
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            
            for data, target in test_loader:
                out = layer(data)
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                loss = (target_onehot - out).pow(2).mean().item()
                total_loss += loss * data.size(0)
                total_v_mean += layer.membrane_potential.mean().item() * data.size(0)
                pred = out.argmax(dim=1)
                test_correct += (pred == target).float().sum().item()
                    
        final_acc = test_correct / TEST_SAMPLES * 100
        avg_loss = total_loss / TEST_SAMPLES
        avg_v = total_v_mean / TEST_SAMPLES
        
        status = "Robust" if final_acc > 90.0 else "Weak"
        if final_acc > 98.0: status = "Excellent"
        if noise >= 0.5:
            if final_acc < 15.0: status = "Theoretical Limit (OK)"
            else: status = "Suspiciously High"
        if noise == 0.45 and final_acc > 88.0: status = "State-of-the-Art"
        
        print(f"{noise:<6.2f} | {final_acc:6.1f}% | {avg_loss:6.4f} | {avg_v:6.3f} | {status}")
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()
