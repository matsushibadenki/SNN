# ファイルパス: snn_research/core/hybrid_core.py
# Title: Phase-Critical Hybrid Neuromorphic Core (BitNet Reservoir Tuned)
# 修正: BitNetリザーバのバイアス有効化とスケーリング調整により、GRPO精度を回復。

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from snn_research.core.layers.logic_gated_snn import PhaseCriticalSCAL
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear

class AdaptiveSparsityLayer(nn.Module):
    """
    Adaptive Top-K with variance-aware sparsity control
    """
    def __init__(self, features: int, base_sparsity: float = 0.10):
        super().__init__()
        self.features = features
        self.base_sparsity = base_sparsity
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor, variance_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward with adaptive sparsity
        Args:
            x: 入力 [batch, features]
            variance_signal: 分散シグナル [features] (オプション)
        """
        x = self.norm(x)

        # 分散に応じてスパースネスを調整
        adaptive_sparsity: float
        if variance_signal is not None:
            # 分散が大きい → より多くのニューロンを活性化
            avg_variance = variance_signal.mean().clamp(0.1, 2.0).item()
            adaptive_sparsity = self.base_sparsity * (1.0 + 0.5 * (avg_variance - 1.0))
            adaptive_sparsity = max(0.05, min(0.25, adaptive_sparsity))
        else:
            adaptive_sparsity = self.base_sparsity

        k = max(1, int(self.features * adaptive_sparsity))

        # Top-K selection
        topk_vals, topk_indices = torch.topk(x, k, dim=1)

        # Sparse activation
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_indices, 1.0)

        return x * mask

class PhaseCriticalHybridCore(nn.Module):
    """
    Hybrid Neuromorphic Core with Phase-Critical SCAL & BitNet Optimization
    
    Architecture:
        Input → Fast Processing (BitNet Reservoir) → Deep Processing (Adaptive Sparsity) 
              → Output (Phase-Critical SCAL)
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int,
        # Reservoir params
        reservoir_mode: str = 'simple',
        # Phase-Critical params
        gamma: float = 0.015, 
        v_th_init: float = 0.5,
        target_spike_rate: float = 0.15
    ):
        super().__init__()
        
        self.reservoir_mode = reservoir_mode
        self.fast_process: nn.Module
        
        # Fast processing layer (Reservoir) with Energy Optimization
        if reservoir_mode == 'phase_critical':
            self.fast_process = PhaseCriticalSCAL(
                in_features, hidden_features, 
                mode='reservoir',
                gamma=gamma * 0.5,
                v_th_init=0.3,
                target_spike_rate=0.20
            )
        else:
            # [Optimization & Fix] BitSpikeLinearによる1.58bit化
            # GRPOの学習（リザーバ計算）を安定させるため、バイアスを有効化し
            # 非線形性を確保。量子化ノイズによる情報損失を補う。
            self.fast_process = nn.Sequential(
                BitSpikeLinear(in_features, hidden_features, bias=True, quantize_inference=True),
                nn.ReLU(),
                # 追加: リザーバの出力を正規化し、次層への信号強度を安定化
                nn.LayerNorm(hidden_features) 
            )
            
        # Deep processing layer (Adaptive sparsity)
        self.deep_process = AdaptiveSparsityLayer(
            hidden_features, base_sparsity=0.15
        )
        
        # Output layer (Phase-Critical SCAL)
        self.output_gate = PhaseCriticalSCAL(
            hidden_features, out_features, 
            mode='readout',
            gamma=gamma,
            v_th_init=v_th_init,
            target_spike_rate=target_spike_rate
        )

    def reset_state(self):
        """全層の状態リセット"""
        if self.reservoir_mode == 'phase_critical':
             if hasattr(self.fast_process, 'reset_state'):
                self.fast_process.reset_state() # type: ignore
        self.output_gate.reset_state()

    def forward(self, x_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Returns:
            Dict with keys: 'output', 'spikes', 'reservoir', 'deep'
        """
        # Fast processing
        if self.reservoir_mode == 'phase_critical':
            fast_result = self.fast_process(x_input) # type: ignore
            reservoir = fast_result['output']
            variance_signal = self.fast_process.class_variance_memory # type: ignore
        else:
            reservoir = self.fast_process(x_input)
            variance_signal = None
            
        # Deep processing
        deep = self.deep_process(reservoir, variance_signal)
        
        # Output with phase-critical dynamics
        output_result = self.output_gate(deep)
        
        return {
            'output': output_result['output'],
            'spikes': output_result['spikes'],
            'reservoir': reservoir,
            'deep': deep,
            'membrane_potential': output_result['membrane_potential'],
            'spike_prob': output_result['spike_prob']
        }

    def update_plasticity(
        self, 
        x_input: torch.Tensor, 
        target: torch.Tensor, 
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        End-to-end plasticity update
        """
        # Forward
        result = self.forward(x_input)
        output = result['output']
        deep = result['deep']
        
        # Target preparation
        target_onehot = torch.zeros_like(output)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # Loss
        loss = torch.nn.functional.mse_loss(output, target_onehot)
        
        # Accuracy
        pred = output.argmax(dim=1)
        accuracy = (pred == target).float().mean()
        
        # Update output layer
        self.output_gate.update_plasticity(deep, result, target, learning_rate)
        
        # Update reservoir if phase-critical
        if self.reservoir_mode == 'phase_critical':
            pass
            
        # Metrics
        metrics = self.output_gate.get_phase_critical_metrics()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'spike_rate': metrics['spike_rate'],
            'mean_threshold': metrics['mean_threshold'],
            'mean_variance': metrics['mean_variance'],
            'reservoir_activity': result['reservoir'].mean().item(),
            'deep_activity': result['deep'].mean().item()
        }

    def get_comprehensive_metrics(self) -> Dict[str, float]:
        """全層の詳細メトリクスを取得"""
        metrics = self.output_gate.get_phase_critical_metrics()
        
        if self.reservoir_mode == 'phase_critical':
            reservoir_metrics = self.fast_process.get_phase_critical_metrics() # type: ignore
            metrics.update({
                f'reservoir_{k}': v for k, v in reservoir_metrics.items()
            })
            
        return metrics

# Legacy wrapper for backward compatibility
class HybridNeuromorphicCore(nn.Module):
    """Backward compatible wrapper"""
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        
        self.core = PhaseCriticalHybridCore(
            in_features, hidden_features, out_features,
            reservoir_mode='simple',
            gamma=0.015,
            v_th_init=0.5
        )
        
    @property
    def membrane_potential(self):
        return self.core.output_gate.membrane_potential

    @property
    def fast_process(self):
        return self.core.fast_process
        
    @property
    def deep_process(self):
        return self.core.deep_process
        
    @property
    def output_gate(self):
        return self.core.output_gate

    def reset_state(self):
        self.core.reset_state()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        result = self.core(x_input)
        return result['output']

    def autonomous_step(
        self, 
        x_input: torch.Tensor, 
        target: Optional[torch.Tensor] = None, 
        learning_rate: float = 0.05
    ) -> Dict[str, float]:
        """Legacy autonomous step interface"""
        
        if target is None:
            # 推論のみ
            result = self.core.forward(x_input)
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'res_density': result['reservoir'].mean().item(),
                'out_density': result['output'].mean().item(),
                'out_v_mean': result['membrane_potential'].mean().item(),
                'out_v_max': result['membrane_potential'].max().item()
            }
        else:
            # 学習ステップ
            metrics = self.core.update_plasticity(x_input, target, learning_rate)
            
            return {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'res_density': metrics['reservoir_activity'],
                'out_density': metrics['deep_activity'],
                'out_v_mean': metrics['mean_threshold'],
                'out_v_max': metrics['mean_threshold'] # 簡略化
            }