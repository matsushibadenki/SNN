# ファイルパス: snn_research/core/ensemble_scal.py
# タイトル: Ensemble SCAL v3.1 (Fix Type Hints)
# 内容: forwardの戻り値の型ヒントを Dict[str, Any] に修正

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from snn_research.core.layers.logic_gated_snn_v2_1 import SCALPerceptionLayer

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
            gains = [40.0, 45.0, 50.0, 55.0, 60.0]
            betas = [0.85, 0.88, 0.90, 0.92, 0.95]
            
            for i in range(n_models):
                idx = i % 5
                gain = gains[idx]
                v_th_init = gain * 0.5 
                
                model = SCALPerceptionLayer(
                    in_features, out_features,
                    time_steps=10,
                    gain=gain,
                    beta_membrane=betas[idx],
                    v_th_init=v_th_init,
                    v_th_min=5.0,
                    v_th_max=gain * 2.0,
                    gamma_th=0.01,
                    target_spike_rate=0.15
                )
                self.models.append(model)
        else:
            for i in range(n_models):
                model = SCALPerceptionLayer(
                    in_features, out_features,
                    time_steps=10,
                    gain=50.0,
                    beta_membrane=0.9,
                    v_th_init=25.0,
                    v_th_min=5.0,
                    v_th_max=100.0,
                    gamma_th=0.01
                )
                self.models.append(model)
    
    def reset_state(self):
        for model in self.models:
            model.reset_state()
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        individual_outputs = []
        individual_spikes = []
        
        for model in self.models:
            result = model(x)
            individual_outputs.append(result['output'])
            individual_spikes.append(result['spikes'])
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        # 単純平均
        ensemble_output = outputs_stacked.mean(dim=0)
        ensemble_spikes = torch.stack(individual_spikes, dim=0).mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': ensemble_spikes,
            'individual_outputs': individual_outputs
        }
    
    def update_plasticity(self, pre, post, target, lr=0.01):
        for model in self.models:
            res = model(pre) # 内部状態更新
            model.update_plasticity(pre, res, target, lr)
            
    def get_ensemble_metrics(self):
        metrics = [m.get_phase_critical_metrics() for m in self.models]
        return {
            'spike_rate_mean': sum(m['spike_rate'] for m in metrics)/len(metrics),
            'mean_threshold_mean': sum(m['mean_threshold'] for m in metrics)/len(metrics),
            'temperature_mean': 1.0
        }

class AdaptiveEnsembleSCAL(EnsembleSCAL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_weights = nn.Parameter(torch.ones(self.n_models))
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        individual_outputs = []
        individual_spikes = []
        
        for model in self.models:
            result = model(x)
            individual_outputs.append(result['output'])
            individual_spikes.append(result['spikes'])
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        
        # 学習可能な重みによる平均
        weights = F.softmax(self.model_weights, dim=0).unsqueeze(1).unsqueeze(2)
        ensemble_output = (outputs_stacked * weights).sum(dim=0)
        
        ensemble_spikes = torch.stack(individual_spikes, dim=0).mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': ensemble_spikes,
            'individual_outputs': individual_outputs,
            'model_weights': self.model_weights.detach()
        }

class BootstrapEnsembleSCAL(nn.Module):
    """
    バギング的なアンサンブル
    """
    def __init__(self, in_features, out_features, n_models=5, bootstrap_ratio=0.8):
        super().__init__()
        self.models = nn.ModuleList([
            SCALPerceptionLayer(
                in_features, out_features,
                time_steps=10,
                gain=50.0,
                beta_membrane=0.9,
                v_th_init=25.0,
                v_th_min=5.0,
                v_th_max=100.0,
                gamma_th=0.01
            ) for _ in range(n_models)
        ])
    
    def reset_state(self):
        for model in self.models:
            model.reset_state()

    def forward(self, x) -> Dict[str, Any]:
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
    
    def update_plasticity(self, pre, post, target, lr=0.01):
        for model in self.models:
            res = model(pre)
            model.update_plasticity(pre, res, target, lr)
            
    def get_ensemble_metrics(self):
        return self.models[0].get_phase_critical_metrics()