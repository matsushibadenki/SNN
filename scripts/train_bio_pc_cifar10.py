# ファイルパス: scripts/train_bio_pc_cifar10.py
# Title: Bio-PCNet 非勾配学習ベンチマーク (Hard k-WTA版)
# Description:
#   絶対値ベースのHard k-WTAにより、L1層のスパース性を強制的に5%に保つ。
#   修正 (v20):
#     - Sparsity: 0.05 (上位5%)
#     - Input Gain: 3.0 (維持)
#   修正 (mypy):
#     - torchvisionの型チェックをスキップするため type: ignore コメントを追加

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]
import logging
import sys
import os
import time
from typing import List, Dict, Any

sys.path.append(os.path.abspath("."))
from snn_research.core.networks.bio_pc_network import BioPCNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("BioPC_Benchmark")

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def flatten_func(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x)

def main() -> None:
    DATASET_NAME = "MNIST" 
    BATCH_SIZE = 64
    EPOCHS = 15
    TIME_STEPS = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3
    
    # 修正: 上位5%のみ
    SPARSITY = 0.05 
    
    INPUT_GAIN = 3.0     
    DEVICE = get_device()
    
    if DATASET_NAME == "MNIST":
        INPUT_DIM = 28 * 28
        LAYER_DIMS = [INPUT_DIM, 512, 256, 10] 
    else:
        INPUT_DIM = 3 * 32 * 32
        LAYER_DIMS = [INPUT_DIM, 1024, 512, 10]

    NEURON_CONFIG: Dict[str, Any] = {
        "type": "lif",
        "tau_mem": 5.0,        
        "base_threshold": 1.0, 
        "adaptation_strength": 0.0, 
        "threshold_decay": 1.0
    }

    logger.info(f"🚀 Starting Bio-PCNet {DATASET_NAME} (Hard k-WTA)")
    logger.info(f"   Device: {DEVICE}, Sparsity: {SPARSITY}")
    logger.info(f"   Layers: {LAYER_DIMS}, Input Gain: {INPUT_GAIN}")

    transform_list = [transforms.ToTensor(), transforms.Lambda(flatten_func), transforms.Lambda(lambda x: x * INPUT_GAIN)]
    transform = transforms.Compose(transform_list)
    data_root = './data'
    os.makedirs(data_root, exist_ok=True)

    if DATASET_NAME == "MNIST":
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BioPCNetwork(
        layer_dims=LAYER_DIMS,
        time_steps=TIME_STEPS,
        neuron_config=NEURON_CONFIG,
        learning_rate=LEARNING_RATE,
        sparsity=SPARSITY
    ).to(DEVICE)
    
    for rule in model.learning_rules:
        rule.hparams['weight_decay'] = WEIGHT_DECAY

    current_lr = LEARNING_RATE
    
    for epoch in range(EPOCHS):
        model.train()
        total_update_mag = 0.0
        start_time = time.time()
        
        if epoch > 0 and epoch % 5 == 0:
            current_lr *= 0.5
            for rule in model.learning_rules:
                rule.hparams['learning_rate'] = current_lr
            logger.info(f"📉 Learning Rate decayed to {current_lr:.6f}")

        state_accum = [0.0] * (len(LAYER_DIMS) - 1)
        active_accum = [0.0] * (len(LAYER_DIMS) - 1)
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            target_onehot = F.one_hot(target, num_classes=10).float() * 0.9 + 0.05
            target_onehot = target_onehot.to(DEVICE)
            
            model.reset_state()
            _ = model(data, targets=target_onehot)
            
            for i, state in enumerate(model.layer_states):
                state_accum[i] += state.mean().item()
                active_accum[i] += (state > 0.01).float().mean().item()
            num_batches += 1
            
            metrics = model.run_learning_step(inputs=data, targets=target_onehot)
            batch_update = sum(v.item() for k, v in metrics.items() if 'update_magnitude' in k)
            total_update_mag += batch_update

            if batch_idx % 200 == 0:
                means = [f"L{i}:{state_accum[i]/num_batches:.3f}" for i in range(len(state_accum))]
                logger.info(f"  Ep {epoch+1} [{batch_idx}] Upd: {batch_update:.4f} | Mean: {', '.join(means)}")

        epoch_time = time.time() - start_time
        actives = [f"L{i}:{active_accum[i]/num_batches:.1%}" for i in range(len(active_accum))]
        logger.info(f"Ep {epoch+1} Done ({epoch_time:.1f}s). Active Rates: {', '.join(actives)}")
        logger.info(f"Total Update Mag: {total_update_mag:.2f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                model.reset_state()
                output = model(data) 
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        logger.info(f"📊 Eval Accuracy: {accuracy:.2f}%")
        
        if accuracy > 90.0:
            logger.info("🎉 High Accuracy Reached!")
            break

if __name__ == "__main__":
    main()