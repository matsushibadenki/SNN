# ファイルパス: snn_research/core/ensemble_scal.py
# タイトル: Ensemble SCAL - 多様性強化版
# 内容: メンバーモデルのパラメータ分散を拡大

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from snn_research.core.layers.logic_gated_snn_v2_1 import ImprovedPhaseCriticalSCAL

class EnsembleSCAL(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_models: int = 5,
        diversity_strategy: str = 'hyperparameter',
        aggregation: str = 'soft_vote'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_models = n_models
        self.aggregation = aggregation
        self.models = nn.ModuleList()
        
        if diversity_strategy == 'hyperparameter':
            # パラメータのばらつきを大きくして、視点を変える
            gammas = [0.003, 0.005, 0.007, 0.009, 0.011]
            # 閾値初期値も分散させる (感度の違いを作る)
            v_th_inits = [0.5, 0.8, 1.0, 1.2, 1.5]
            
            for i in range(n_models):
                idx = i % 5
                model = ImprovedPhaseCriticalSCAL(
                    in_features, out_features,
                    mode='readout',
                    gamma=gammas[idx],
                    v_th_init=v_th_inits[idx],
                    v_th_max=20.0,
                    use_multiscale=True,
                    # 制御強度も変える
                    spike_rate_control_strength=0.08 + (i * 0.01) 
                )
                self.models.append(model)
        else:
            for i in range(n_models):
                model = ImprovedPhaseCriticalSCAL(
                    in_features, out_features,
                    mode='readout',
                    gamma=0.005,
                    v_th_init=1.0,
                    v_th_max=20.0,
                    use_multiscale=True,
                    spike_rate_control_strength=0.1
                )
                self.models.append(model)
    
    def reset_state(self):
        for model in self.models:
            model.reset_state()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        individual_outputs = []
        individual_spikes = []
        for model in self.models:
            result = model(x)
            individual_outputs.append(result['output'])
            individual_spikes.append(result['spikes'])
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        ensemble_output = outputs_stacked.mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': torch.stack(individual_spikes, dim=0).mean(dim=0),
            'individual_outputs': individual_outputs
        }
    
    def update_plasticity(self, pre, post, target, lr=0.02):
        for model in self.models:
            res = model(pre)
            model.update_plasticity(pre, res, target, lr)
            
    def get_ensemble_metrics(self):
        metrics = [m.get_phase_critical_metrics() for m in self.models]
        return {
            'spike_rate_mean': sum(m['spike_rate'] for m in metrics)/len(metrics),
            'mean_threshold_mean': sum(m['mean_threshold'] for m in metrics)/len(metrics),
            'temperature_mean': sum(m['temperature'] for m in metrics)/len(metrics)
        }

class AdaptiveEnsembleSCAL(EnsembleSCAL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_weights = nn.Parameter(torch.ones(self.n_models))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        individual_outputs = []
        individual_spikes = []
        individual_confidences = []
        
        for model in self.models:
            result = model(x)
            output = result['output']
            spikes = result['spikes']
            
            individual_outputs.append(output)
            individual_spikes.append(spikes)
            
            entropy = -(output * (output + 1e-8).log()).sum(dim=1)
            confidence = 1.0 / (entropy + 1e-8)
            individual_confidences.append(confidence)
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        confidences_stacked = torch.stack(individual_confidences, dim=0)
        
        weights = F.softmax(self.model_weights, dim=0).unsqueeze(1).unsqueeze(2)
        confidences_norm = F.softmax(confidences_stacked, dim=0).unsqueeze(2)
        
        combined_weights = weights * confidences_norm
        combined_weights = combined_weights / combined_weights.sum(dim=0, keepdim=True)
        
        ensemble_output = (outputs_stacked * combined_weights).sum(dim=0)
        ensemble_spikes = torch.stack(individual_spikes, dim=0).mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': ensemble_spikes,
            'individual_outputs': individual_outputs,
            'model_weights': self.model_weights.detach()
        }

class BootstrapEnsembleSCAL(nn.Module):
    def __init__(self, in_features, out_features, n_models=5, bootstrap_ratio=0.8):
        super().__init__()
        self.n_models = n_models
        self.bootstrap_ratio = bootstrap_ratio
        self.models = nn.ModuleList([
            ImprovedPhaseCriticalSCAL(
                in_features, out_features,
                mode='readout', gamma=0.008, v_th_init=3.0, v_th_max=20.0, use_multiscale=True
            ) for _ in range(n_models)
        ])
    
    def forward(self, x):
        outputs = []
        spikes = []
        for m in self.models:
            res = m(x)
            outputs.append(res['output'])
            spikes.append(res['spikes'])
        
        return {
            'output': torch.stack(outputs).mean(dim=0),
            'spikes': torch.stack(spikes).mean(dim=0),
            'individual_outputs': outputs
        }
    
    def update_plasticity(self, pre, post, target, lr=0.02):
        for model in self.models:
            res = model(pre)
            model.update_plasticity(pre, res, target, lr)
            
    def get_ensemble_metrics(self):
        return self.models[0].get_phase_critical_metrics()