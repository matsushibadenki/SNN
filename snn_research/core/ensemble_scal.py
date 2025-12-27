# ファイルパス: snn_research/core/ensemble_scal.py
# タイトル: Ensemble SCAL - 多様性による性能向上
# 内容: 複数のSCALモデルを統合し、ノイズ耐性を強化

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class EnsembleSCAL(nn.Module):
    """
    Ensemble of Phase-Critical SCAL models
    
    戦略:
    1. 異なる初期化で複数モデルを訓練
    2. 異なるハイパーパラメータで多様性を確保
    3. 投票または確率平均で統合
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_models: int = 5,
        diversity_strategy: str = 'hyperparameter',  # 'hyperparameter' or 'initialization'
        aggregation: str = 'soft_vote'  # 'soft_vote' or 'hard_vote'
    ):
        super().__init__()
        
        from snn_research.core.layers.logic_gated_snn_v2_1 import ImprovedPhaseCriticalSCAL
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_models = n_models
        self.aggregation = aggregation
        
        self.models = nn.ModuleList()
        
        if diversity_strategy == 'hyperparameter':
            # 異なるハイパーパラメータで多様性
            gammas = [0.005, 0.008, 0.010, 0.012, 0.015][:n_models]
            v_th_inits = [0.6, 0.7, 0.8, 0.9, 1.0][:n_models]
            use_multiscales = [False, False, True, True, False][:n_models]
            
            for i in range(n_models):
                model = ImprovedPhaseCriticalSCAL(
                    in_features, out_features,
                    mode='readout',
                    gamma=gammas[i],
                    v_th_init=v_th_inits[i],
                    v_th_min=0.3,
                    use_multiscale=use_multiscales[i]
                )
                self.models.append(model)
        
        else:  # 'initialization'
            # 同じハイパーパラメータ、異なる初期化
            for i in range(n_models):
                model = ImprovedPhaseCriticalSCAL(
                    in_features, out_features,
                    mode='readout',
                    gamma=0.008,
                    v_th_init=0.8,
                    use_multiscale=(i % 2 == 0)  # 交互にマルチスケール
                )
                self.models.append(model)
    
    def reset_state(self):
        for model in self.models:
            model.reset_state()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ensemble forward pass
        
        Returns:
            Dict with 'output', 'spikes', 'individual_outputs'
        """
        batch_size = x.size(0)
        
        # 各モデルの出力を収集
        individual_outputs = []
        individual_spikes = []
        individual_probs = []
        
        for model in self.models:
            result = model(x)
            individual_outputs.append(result['output'])
            individual_spikes.append(result['spikes'])
            individual_probs.append(result['spike_prob'])
        
        # Stack
        outputs_stacked = torch.stack(individual_outputs, dim=0)  # [n_models, batch, classes]
        spikes_stacked = torch.stack(individual_spikes, dim=0)
        
        # Aggregation
        if self.aggregation == 'soft_vote':
            # 確率平均
            ensemble_output = outputs_stacked.mean(dim=0)
        else:  # 'hard_vote'
            # 多数決
            predictions = outputs_stacked.argmax(dim=2)  # [n_models, batch]
            ensemble_output = torch.zeros(batch_size, self.out_features, device=x.device)
            for i in range(batch_size):
                votes = predictions[:, i]
                # 最多得票クラス
                winner = torch.mode(votes)[0]
                ensemble_output[i, winner] = 1.0
        
        # Ensemble spikes (average)
        ensemble_spikes = spikes_stacked.mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': ensemble_spikes,
            'individual_outputs': individual_outputs,
            'individual_spikes': individual_spikes
        }
    
    def update_plasticity(
        self,
        pre_activity: torch.Tensor,
        post_output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        learning_rate: float = 0.02
    ):
        """各モデルを独立に更新"""
        for model in self.models:
            result = model(pre_activity)
            model.update_plasticity(pre_activity, result, target, learning_rate)
    
    def get_ensemble_metrics(self) -> Dict[str, float]:
        """アンサンブル全体のメトリクス"""
        metrics_list = [model.get_phase_critical_metrics() for model in self.models]
        
        # 平均と標準偏差
        ensemble_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            ensemble_metrics[f'{key}_mean'] = sum(values) / len(values)
            ensemble_metrics[f'{key}_std'] = torch.tensor(values).std().item()
        
        return ensemble_metrics


class AdaptiveEnsembleSCAL(EnsembleSCAL):
    """
    Adaptive Ensemble with confidence weighting
    
    各モデルの確信度に応じて重み付け
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # モデルごとの重み（学習可能）
        self.model_weights = nn.Parameter(torch.ones(self.n_models))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # 各モデルの出力
        individual_outputs = []
        individual_confidences = []
        
        for model in self.models:
            result = model(x)
            output = result['output']
            individual_outputs.append(output)
            
            # 確信度: エントロピーの逆数
            entropy = -(output * (output + 1e-8).log()).sum(dim=1)
            confidence = 1.0 / (entropy + 1e-8)
            individual_confidences.append(confidence)
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        confidences_stacked = torch.stack(individual_confidences, dim=0)
        
        # モデル重みとconfidenceを統合
        weights = F.softmax(self.model_weights, dim=0).unsqueeze(1).unsqueeze(2)
        confidences_norm = F.softmax(confidences_stacked, dim=0).unsqueeze(2)
        
        combined_weights = weights * confidences_norm
        combined_weights = combined_weights / combined_weights.sum(dim=0, keepdim=True)
        
        # 重み付き平均
        ensemble_output = (outputs_stacked * combined_weights).sum(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': torch.zeros_like(ensemble_output),  # Placeholder
            'individual_outputs': individual_outputs,
            'model_weights': self.model_weights.detach()
        }


class BootstrapEnsembleSCAL(nn.Module):
    """
    Bootstrap Aggregating (Bagging) for SCAL
    
    各モデルを異なるデータサブセットで訓練
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_models: int = 5,
        bootstrap_ratio: float = 0.8
    ):
        super().__init__()
        
        from snn_research.core.layers.logic_gated_snn_v2_1 import ImprovedPhaseCriticalSCAL
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_models = n_models
        self.bootstrap_ratio = bootstrap_ratio
        
        self.models = nn.ModuleList([
            ImprovedPhaseCriticalSCAL(
                in_features, out_features,
                mode='readout',
                gamma=0.008,
                v_th_init=0.8,
                use_multiscale=(i % 2 == 0)
            )
            for i in range(n_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """同じforward処理"""
        individual_outputs = []
        
        for model in self.models:
            result = model(x)
            individual_outputs.append(result['output'])
        
        outputs_stacked = torch.stack(individual_outputs, dim=0)
        ensemble_output = outputs_stacked.mean(dim=0)
        
        return {
            'output': ensemble_output,
            'spikes': torch.zeros_like(ensemble_output),
            'individual_outputs': individual_outputs
        }
    
    def update_plasticity_bootstrap(
        self,
        pre_activity: torch.Tensor,
        target: torch.Tensor,
        learning_rate: float = 0.02
    ):
        """
        Bootstrap訓練: 各モデルに異なるサブセットを使用
        """
        batch_size = pre_activity.size(0)
        subsample_size = int(batch_size * self.bootstrap_ratio)
        
        for model in self.models:
            # ランダムサブサンプル（復元抽出）
            indices = torch.randint(0, batch_size, (subsample_size,), device=pre_activity.device)
            
            sub_pre = pre_activity[indices]
            sub_target = target[indices]
            
            result = model(sub_pre)
            model.update_plasticity(sub_pre, result, sub_target, learning_rate)