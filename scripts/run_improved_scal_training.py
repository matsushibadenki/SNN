# ファイルパス: scripts/run_improved_scal_training.py
# タイトル: SCAL v2.1 改善版トレーニング (Sparse Projection)
# 内容: Pattern Separatorを用いた高精度トレーニング設定

import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np
from typing import Tuple, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def generate_synthetic_data(
    num_samples: int,
    in_features: int,
    out_features: int,
    prototypes: torch.Tensor,
    noise_level: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = prototypes.device
    labels = torch.randint(0, out_features, (num_samples,), device=device)
    patterns = prototypes[labels]
    noise_mask = (torch.rand(num_samples, in_features, device=device) < noise_level).float()
    x = torch.abs(patterns - noise_mask)
    return x, labels

def run_experiment(use_ensemble: bool = False, use_multiscale: bool = True):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}\n")
    
    IN_FEATURES = 784
    OUT_FEATURES = 10
    TRAIN_SAMPLES = 50000
    TEST_SAMPLES = 10000
    
    # 集中トレーニング設定
    curriculum = [
        {'noise': 0.10, 'epochs': 5, 'lr': 0.04, 'batch': 2048},
        {'noise': 0.30, 'epochs': 5, 'lr': 0.03, 'batch': 4096},
        {'noise': 0.40, 'epochs': 10, 'lr': 0.02, 'batch': 8192},
        # Noise 0.45 での微調整期間を設ける
        {'noise': 0.45, 'epochs': 60, 'lr': 0.015, 'batch': 8192}, 
        {'noise': 0.48, 'epochs': 30, 'lr': 0.008, 'batch': 8192},
    ]
    
    if use_ensemble:
        # AdaptiveEnsembleSCAL を使用
        from snn_research.core.ensemble_scal import AdaptiveEnsembleSCAL
        print("=== Adaptive Ensemble SCAL (Spatial Denoising) ===")
        model = AdaptiveEnsembleSCAL(
            IN_FEATURES, OUT_FEATURES,
            n_models=5,
            diversity_strategy='hyperparameter',
            aggregation='soft_vote'
        ).to(device)
    else:
        from snn_research.core.layers.logic_gated_snn_v2_1 import ImprovedPhaseCriticalSCAL
        print("=== Improved Phase-Critical SCAL v2.1 (Pattern Separation) ===")
        model = ImprovedPhaseCriticalSCAL(
            IN_FEATURES, OUT_FEATURES,
            mode='readout',
            gamma=0.008,
            v_th_init=1.0,
            v_th_min=0.1,
            v_th_max=10.0,
            target_spike_rate=0.15,
            use_multiscale=use_multiscale,
            spike_rate_control_strength=0.05
        ).to(device)
    
    print(f"Multiscale features: {'Enabled' if use_multiscale else 'Disabled'}")
    print(f"Ensemble: {'Enabled' if use_ensemble else 'Disabled'}\n")
    
    prototypes = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) > 0.0).float()
    
    print(f"{'Epoch':<6} | {'Stage':<20} | {'Acc':<7} | {'Loss':<7} | "
          f"{'SpkRate':<8} | {'V_th':<7} | {'Temp':<7} | {'Time'}")
    print("-" * 95)
    
    start_time = time.time()
    epoch_counter = 0
    
    for stage in curriculum:
        noise_level = stage['noise']
        stage_epochs = stage['epochs']
        learning_rate = stage['lr']
        batch_size = stage['batch']
        
        stage_name = f"Noise {noise_level:.2f}"
        
        for ep in range(stage_epochs):
            epoch_counter += 1
            
            x_train, y_train = generate_synthetic_data(
                TRAIN_SAMPLES, IN_FEATURES, OUT_FEATURES,
                prototypes, noise_level
            )
            
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            
            for batch_x, batch_y in loader:
                result = model(batch_x)
                output = result['output']
                
                target_onehot = torch.zeros_like(output)
                target_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                loss = F.mse_loss(output, target_onehot)
                
                pred = output.argmax(dim=1)
                correct = (pred == batch_y).sum().item()
                
                model.update_plasticity(batch_x, result, batch_y, learning_rate)
                
                epoch_loss += loss.item() * batch_x.size(0)
                epoch_correct += correct
                total_samples += batch_x.size(0)
            
            avg_loss = epoch_loss / total_samples
            accuracy = epoch_correct / total_samples
            
            if use_ensemble:
                metrics = model.get_ensemble_metrics()
                spike_rate = metrics['spike_rate_mean']
                v_th = metrics['mean_threshold_mean']
                temp = metrics['temperature_mean']
            else:
                metrics = model.get_phase_critical_metrics()
                spike_rate = metrics['spike_rate']
                v_th = metrics['mean_threshold']
                temp = metrics['temperature']
            
            elapsed = time.time() - start_time
            
            if epoch_counter % 5 == 0 or ep == stage_epochs - 1:
                print(f"{epoch_counter:<6} | {stage_name:<20} | "
                      f"{accuracy*100:6.2f}% | {avg_loss:7.5f} | "
                      f"{spike_rate*100:7.2f}% | {v_th:7.4f} | "
                      f"{temp:7.4f} | {elapsed:.0f}s")
    
    print("\n=== Training Complete ===\n")
    
    print("=== Robustness Evaluation (10 trials per noise level) ===")
    print(f"{'Noise':<7} | {'Acc':<12} | {'Loss':<8} | {'SpkRate':<12} | {'Status'}")
    print("-" * 70)
    
    model.eval()
    test_noise_levels = [0.10, 0.20, 0.30, 0.40, 0.45, 0.48, 0.50]
    
    for noise in test_noise_levels:
        x_test, y_test = generate_synthetic_data(
            TEST_SAMPLES, IN_FEATURES, OUT_FEATURES,
            prototypes, noise
        )
        
        with torch.no_grad():
            n_trials = 10
            accuracies = []
            losses = []
            spike_rates = []
            
            for _ in range(n_trials):
                test_dataset = TensorDataset(x_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=4096)
                
                correct = 0
                total_loss = 0.0
                total_spikes = 0.0
                total = 0
                
                for batch_x, batch_y in test_loader:
                    result = model(batch_x)
                    output = result['output']
                    spikes = result['spikes']
                    
                    target_onehot = torch.zeros_like(output)
                    target_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                    
                    loss = F.mse_loss(output, target_onehot)
                    pred = output.argmax(dim=1)
                    
                    correct += (pred == batch_y).sum().item()
                    total_loss += loss.item() * batch_x.size(0)
                    total_spikes += spikes.mean().item() * batch_x.size(0)
                    total += batch_x.size(0)
                
                accuracies.append(correct / total)
                losses.append(total_loss / total)
                spike_rates.append(total_spikes / total)
        
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        loss_mean = np.mean(losses)
        spike_mean = np.mean(spike_rates)
        
        if acc_mean > 0.98: status = "Excellent"
        elif acc_mean > 0.90: status = "Good"
        elif acc_mean > 0.85: status = "Acceptable"
        elif noise >= 0.48: status = "Near Limit"
        else: status = "Weak"
        
        if noise == 0.45 and acc_mean >= 0.90: status = "TARGET ACHIEVED ✓✓"
        elif noise == 0.45 and acc_mean >= 0.88: status = "SOTA ✓"
        if noise == 0.48 and acc_mean >= 0.42: status = "EXCELLENT (>42%) ✓✓"
        
        print(f"{noise:<7.2f} | {acc_mean*100:5.2f}%±{acc_std*100:4.2f}% | "
              f"{loss_mean:8.5f} | {spike_mean*100:5.2f}%±{np.std(spike_rates)*100:4.2f}% | "
              f"{status}")
    
    print("\n=== Experiment Complete ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--multiscale', action='store_true', default=True, help='Use multiscale features')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SCAL v2.1 - Improved Training")
    print("="*70 + "\n")
    
    run_experiment(use_ensemble=args.ensemble, use_multiscale=args.multiscale)
