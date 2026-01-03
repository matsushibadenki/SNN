# ファイルパス: snn_research/training/trainers/particle_filter.py
# Title: Particle Filter Trainer
# Description: 粒子フィルタを用いた非勾配学習トレーナー。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import copy

from snn_research.models.bio.simple_network import BioSNN

class ParticleFilterTrainer:
    def __init__(self, base_model: BioSNN, config: Dict[str, Any], device: str):
        self.base_model = base_model.to(device)
        self.device = device
        self.config = config
        self.num_particles: int = config['training']['biologically_plausible']['particle_filter']['num_particles']
        self.noise_std: float = config['training']['biologically_plausible']['particle_filter']['noise_std']
        self.particles: List[nn.Module] = [copy.deepcopy(self.base_model) for _ in range(self.num_particles)]
        self.particle_weights = torch.ones(self.num_particles, device=self.device) / self.num_particles
        print(f"🌪️ ParticleFilterTrainer initialized with {self.num_particles} particles.")
        
    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        # パラメータにノイズを注入（探索ステップ）
        for particle in self.particles:
            with torch.no_grad():
                for param in particle.parameters(): 
                    param.add_(torch.randn_like(param) * self.noise_std)
        
        log_likelihoods: List[float] = []
        for particle in self.particles:
            particle.eval()
            with torch.no_grad():
                # 修正: BioSNNは[Batch, Input]の形状を期待するため、バッチ次元を保持または追加する
                # dataが[1, N]の場合はそのまま、[N]の場合は[1, N]にunsqueezeする
                input_tensor = data
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                
                probs = torch.clamp(input_tensor, 0.0, 1.0)
                input_spikes = torch.bernoulli(probs)
                outputs, _ = particle(input_spikes) # type: ignore[operator]
                
                if targets is not None:
                     # ターゲットも出力の次元に合わせる (outputs: [Batch, Output])
                     target_tensor = targets
                     if target_tensor.dim() == 1:
                         target_tensor = target_tensor.unsqueeze(0)
                     
                     loss = F.mse_loss(outputs, target_tensor)
                     log_likelihoods.append(-loss.item())
                else: 
                    log_likelihoods.append(0.0)
        
        log_likelihoods_tensor = torch.tensor(log_likelihoods, device=self.device)
        # 重みの更新 (尤度に基づく)
        self.particle_weights *= torch.exp(log_likelihoods_tensor - log_likelihoods_tensor.max())
        
        if self.particle_weights.sum() > 0: 
            self.particle_weights /= self.particle_weights.sum()
        else: 
            self.particle_weights.fill_(1.0 / self.num_particles)
            
        # リサンプリング (Effective Sample Sizeに基づく)
        if 1.0 / (self.particle_weights**2).sum() < self.num_particles / 2.0:
            indices = torch.multinomial(self.particle_weights, self.num_particles, replacement=True)
            self.particles = [copy.deepcopy(self.particles[i]) for i in indices]
            self.particle_weights.fill_(1.0 / self.num_particles)
            
        return -log_likelihoods_tensor.max().item()