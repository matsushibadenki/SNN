# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: Quartic Contrast & Safe Curriculum)
# 内容: M4/MPS最適化、軽量化(4096 dim)、4乗コントラスト、ノイズ上限ガード付きカリキュラム

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
    合成データの生成 (XOR Noise) - 高速化版 (Vectorized)
    """
    device = prototypes.device if prototypes is not None else torch.device('cpu')

    if prototypes is None:
        # プロトタイプ生成 (0 or 1)
        prototypes = (torch.randn(out_features, in_features, device=device) > 0.0).float()
    
    # ラベルを一括生成
    labels = torch.randint(0, out_features, (num_samples,), device=device)
    
    # 選択されたプロトタイプを取得
    patterns = prototypes[labels]
    
    # ノイズ確率の生成
    if isinstance(noise_level, (tuple, list)):
        # サンプルごとに異なるノイズレベルを一様分布から生成
        probs = torch.rand(num_samples, 1, device=device) * (noise_level[1] - noise_level[0]) + noise_level[0]
    else:
        # 固定ノイズレベル
        probs = torch.full((num_samples, 1), noise_level, device=device)
    
    # ノイズマスクの生成 (確率に基づいてビット反転位置を決定)
    noise_mask = (torch.rand(num_samples, in_features, device=device) < probs).float()
    
    # XOR的なノイズ付加: abs(pattern - noise)
    x = torch.abs(patterns - noise_mask)
    y = labels
    
    return x, y, prototypes

def get_optimal_device():
    """環境に応じた最適なデバイスを選択 (M4 Mac対応)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # macOS Apple Silicon (M1/M2/M3/M4) 用のMetal Performance Shaders
        return torch.device("mps")
    else:
        return torch.device("cpu")

def run_simulation():
    set_seed(42)
    device = get_optimal_device()
    print(f"Running on Device: {device}")

    IN_FEATURES = 784
    # 軽量化: 10000 -> 4096 (十分な表現力を維持しつつ、計算速度を2.5倍向上)
    HIDDEN_FEATURES = 4096 
    OUT_FEATURES = 10
    
    # MPS/GPU活用のためバッチサイズを拡大
    BATCH_SIZE = 512
    # ノイズを平均化するためにサンプル数を増量
    TOTAL_SAMPLES = 30000
    EPOCHS = 40
    
    INITIAL_LR = 0.03
    
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons.")
    print(f"Training Logic: Quartic Contrast (x100), Speed Optimized (MPS), Safe-Limit Curriculum.")
    
    _, _, shared_prototypes = generate_synthetic_data(num_samples=1, in_features=IN_FEATURES, out_features=OUT_FEATURES)
    shared_prototypes = shared_prototypes.to(device)

    print("\nStarting Curriculum Training Phase...")
    print(f"{'Epoch':<6} | {'Noise Range':<15} | {'Acc':<6} | {'Loss':<6} | {'LR':<8} | {'R_Spk%':<6} | {'V_Mean':<6} | {'Time'}")
    print("-" * 96)
    
    start_time = time.time()
    current_lr = INITIAL_LR
    
    for epoch in range(EPOCHS):
        # カリキュラム学習設定 (High-Noise Emphasis)
        # ノイズ0.5は完全ランダム(情報量ゼロ)のため、学習データに含めると重みが破壊される。
        # 上限を0.46 (情報の残存4%) に留め、構造を維持させる。
        if epoch < 8:
            # 基礎学習
            current_noise_range = (0.0, 0.25) 
            current_lr = INITIAL_LR * (0.95 ** epoch)
        elif epoch < 20:
            # 応用学習
            current_noise_range = (0.1, 0.42)
            current_lr = INITIAL_LR * (0.95 ** epoch)
        elif epoch < 32:
            # 高ノイズ適応 (最も難しい領域をカバー)
            current_noise_range = (0.30, 0.46)
            current_lr = 0.005
        else:
            # 極限環境適応 (0.40-0.46の境界領域を反復し、微調整)
            # 0.5は含めない
            current_noise_range = (0.40, 0.46)
            current_lr = 0.002 # 微調整
            
        # データ生成 (高速化版)
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
              f"{current_lr:.6f} | "
              f"{metrics.get('res_density', 0)*100:5.1f}% | "
              f"{metrics.get('out_v_mean', 0):6.3f} | "
              f"{elapsed:.0f}s")
        
    print("Optimization Complete.")
    
    print("\n=== Running Robustness Evaluation (Stress Test) ===")
    core.eval()
    
    # 0.45, 0.48 の粘りを確認
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5]
    TEST_SAMPLES = 5000 
    
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
                if hasattr(core, 'output_gate') and hasattr(core.output_gate, 'membrane_potential'):
                    total_v_mean += core.output_gate.membrane_potential.mean().item() * data.size(0)

                pred = out.argmax(dim=1)
                test_correct += (pred == target).float().sum().item()
                    
        final_acc = test_correct / TEST_SAMPLES * 100
        avg_loss = total_loss / TEST_SAMPLES
        avg_spk = (total_spikes / TEST_SAMPLES) * 100
        avg_v = total_v_mean / TEST_SAMPLES
        
        status = "Robust" if final_acc > 85.0 else "Weak"
        if final_acc > 98.0: status = "Excellent"
        if noise >= 0.5:
            if final_acc < 15.0: status = "Theoretical Limit (OK)"
            else: status = "Suspiciously High"
        
        print(f"{noise:<6.2f} | {final_acc:6.1f}% | {avg_loss:6.4f} | {avg_spk:5.1f}% | {avg_v:6.3f} | {status}")
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    run_simulation()
