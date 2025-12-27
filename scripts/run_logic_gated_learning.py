# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: Hyper-Contrast Boosting)
# 内容: Hyper-Contrast Boostingによる極限コントラスト学習を利用し、Acc 88%の壁を突破する。

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np
from typing import Union, Tuple, List

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
                          mixture_config: List[Tuple[Union[float, Tuple[float, float]], float]] = None):
    
    device = prototypes.device if prototypes is not None else torch.device('cpu')

    if prototypes is None:
        prototypes = (torch.randn(out_features, in_features, device=device) > 0.0).float()
    
    x_list, y_list = [], []
    
    if mixture_config is None:
        mixture_config = [(0.1, 1.0)]
    
    current_count = 0
    for i, (n_level, ratio) in enumerate(mixture_config):
        if i == len(mixture_config) - 1:
            count = num_samples - current_count
        else:
            count = int(num_samples * ratio)
        
        current_count += count
        
        if count > 0:
            labels = torch.randint(0, out_features, (count,), device=device)
            patterns = prototypes[labels]
            
            if isinstance(n_level, (tuple, list)):
                probs = torch.rand(count, 1, device=device) * (n_level[1] - n_level[0]) + n_level[0]
            else:
                probs = torch.full((count, 1), n_level, device=device)
            
            noise_mask = (torch.rand(count, in_features, device=device) < probs).float()
            x = torch.abs(patterns - noise_mask)
            x_list.append(x)
            y_list.append(labels)
    
    if x_list:
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
    else:
        return torch.empty(0), torch.empty(0), prototypes
    
    return x, y, prototypes

def run_simulation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    IN_FEATURES = 784
    OUT_FEATURES = 10
    TOTAL_SAMPLES = 60000
    
    # [修正] カリキュラム: Hyper-Contrast Boosting v24 (Chaotic Resonance)
    # v23の反省: 巨大バッチ固定では局所解にハマる。
    # v24の戦略: 
    # 1. バッチサイズを交互に切り替える(Oscillation)。
    #    8192(安定) <-> 1024(不安定) を繰り返すことで、解空間を揺さぶり、より良い極小値へ誘導する。
    # 2. ターゲット範囲を広め(0.40-0.46)から狭め(0.44-0.455)へ徐々にシフト。
    # 3. サポート(0.35)は維持し、V_Meanを守る。
    curriculum_stages = [
        # 初期: 基礎
        {'mixture': [((0.0, 0.30), 1.0)], 'epochs': 10, 'lr': 0.1, 'batch': 2048},
        {'mixture': [((0.2, 0.40), 1.0)], 'epochs': 10, 'lr': 0.05, 'batch': 2048},
        
        # 中盤: 混合開始
        {'mixture': [((0.35, 0.45), 0.8), (0.30, 0.2)], 'epochs': 20, 'lr': 0.02, 'batch': 4096},
        
        # === Chaotic Resonance Phase ===
        # バッチサイズを振動させ、局所解からの脱出を図る。
        
        # Cycle 1: 広域探索
        {'mixture': [((0.40, 0.46), 0.85), (0.35, 0.15)], 'epochs': 20, 'lr': 0.005, 'batch': 8192},
        {'mixture': [((0.40, 0.46), 0.85), (0.35, 0.15)], 'epochs': 20, 'lr': 0.005, 'batch': 1024}, # 揺さぶり
        
        # Cycle 2: ターゲット絞り込み
        {'mixture': [((0.43, 0.455), 0.9), (0.35, 0.1)], 'epochs': 20, 'lr': 0.002, 'batch': 8192},
        {'mixture': [((0.43, 0.455), 0.9), (0.35, 0.1)], 'epochs': 20, 'lr': 0.002, 'batch': 1024}, # 揺さぶり
        
        # Cycle 3: 最終収束 (ターゲット0.45)
        # ここでは巨大バッチで安定化させつつ、0.455の上限で少し負荷をかける
        {'mixture': [((0.44, 0.455), 0.9), (0.35, 0.1)], 'epochs': 50, 'lr': 0.001, 'batch': 8192},
        
        # Final Polish: 最小LRで仕上げ
        {'mixture': [((0.445, 0.455), 0.9), (0.35, 0.1)], 'epochs': 50, 'lr': 0.0005, 'batch': 4096}, 
    ]
    
    layer = LogicGatedSNN(IN_FEATURES, OUT_FEATURES, mode='readout').to(device)
    
    print(f"\nModel initialized: LogicGatedSNN (Statistical Averaging Mode)")
    print(f"Training Logic: Granular Curriculum Learning (Chaotic Resonance v24).")
    
    _, _, shared_prototypes = generate_synthetic_data(num_samples=1, in_features=IN_FEATURES, out_features=OUT_FEATURES)
    shared_prototypes = shared_prototypes.to(device)

    print("\nStarting Curriculum Training Phase...")
    print(f"{'Epoch':<6} | {'Target Range (Ratio)':<25} | {'Supp':<8} | {'Acc':<6} | {'Loss':<6} | {'LR':<8} | {'Batch':<6} | {'V_Mean':<6} | {'Time'}")
    print("-" * 115)
    
    start_time = time.time()
    
    current_epoch = 0
    
    # カリキュラムステージごとのループ
    for stage in curriculum_stages:
        mixture = stage['mixture']
        stage_epochs = stage['epochs']
        lr = stage['lr']
        batch_size = stage['batch']
        
        # 表示用文字列
        main_conf = mixture[0]
        supp_conf = mixture[1] if len(mixture) > 1 else (0.0, 0.0)
        main_str = f"{str(main_conf[0])} ({main_conf[1]*100:.0f}%)"
        supp_str = f"{str(supp_conf[0])} ({supp_conf[1]*100:.0f}%)" if supp_conf[1] > 0 else "None"
        
        for _ in range(stage_epochs):
            current_epoch += 1
            
            x_train, y_train, _ = generate_synthetic_data(
                num_samples=TOTAL_SAMPLES, 
                in_features=IN_FEATURES, 
                out_features=OUT_FEATURES,
                prototypes=shared_prototypes,
                mixture_config=mixture
            )
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type=='cuda'))

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
            
            print(f"{current_epoch:<6} | {main_str:<25} | {supp_str:<8} | "
                  f"{epoch_acc:5.1f}% | "
                  f"{avg_loss:6.4f} | "
                  f"{lr:.6f} | "
                  f"{batch_size:<6} | "
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
            mixture_config=[(noise, 1.0)]
        )
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        test_correct = 0
        total_loss = 0.0
        total_v_mean = 0.0
        
        with torch.no_grad():
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=4096)
            
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
        # ターゲット判定
        if noise == 0.45:
             if final_acc > 88.0: status = "TARGET ACHIEVED (SOTA)"
             else: status = "Weak (Target Missed)"
        
        print(f"{noise:<6.2f} | {final_acc:6.1f}% | {avg_loss:6.4f} | {avg_v:6.3f} | {status}")
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()