# ファイルパス: snn_research/conversion/bio_calibrator.py
# Title: Deep Bio-Calibrator (HSEO-based SNN Fine-Tuning) - ロジック修正版
# Description:
#   Roadmap v14.0 "Deep Bio-Calibration" の実装。
#   HSEO (Hybrid Swarm Evolution Optimization) を用いて、
#   変換後のSNNモデルのニューロンパラメータ（主に閾値）を
#   キャリブレーションデータに基づいて自動最適化する。
#   修正: ターゲットのOne-Hot判定ロジックを修正し、シーケンスデータ(LongTensor)が
#   誤ってargmaxで次元削減されるバグを解消。

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.core.neurons import (
    AdaptiveLIFNeuron, DualThresholdNeuron, ScaleAndFireNeuron, IzhikevichNeuron
)
from snn_research.optimization.hseo import optimize_with_hseo

logger = logging.getLogger(__name__)

class DeepBioCalibrator:
    """
    SNNモデルの生物学的パラメータ（閾値など）をHSEOを用いて自動校正するクラス。
    """
    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Any,
        device: str = "cpu",
        hseo_particles: int = 10,
        hseo_iterations: int = 15,
        metric_weight_accuracy: float = 1.0,
        metric_weight_sparsity: float = 0.1
    ):
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device
        self.hseo_particles = hseo_particles
        self.hseo_iterations = hseo_iterations
        self.metric_weight_accuracy = metric_weight_accuracy
        self.metric_weight_sparsity = metric_weight_sparsity
        
        # 最適化対象のニューロン層を特定
        self.target_layers: List[nn.Module] = []
        self.layer_names: List[str] = []
        self._identify_target_layers()
        
        logger.info(f"🧬 DeepBioCalibrator initialized. Targets: {len(self.target_layers)} layers.")

    def _identify_target_layers(self) -> None:
        """最適化可能なニューロン層を収集する"""
        for name, module in self.model.named_modules():
            if isinstance(module, (AdaptiveLIFNeuron, DualThresholdNeuron, ScaleAndFireNeuron, IzhikevichNeuron)):
                self.target_layers.append(module)
                self.layer_names.append(name)

    @torch.no_grad()
    def _evaluate_params(self, scaling_factors: np.ndarray) -> float:
        """
        目的関数: 指定されたスケーリング係数を閾値に適用し、スコア（損失）を計算する。
        HSEOは最小化を行うため、(1 - Accuracy) + SparsityPenalty を返す。
        """
        # 1. パラメータの一時的な適用
        original_thresholds = []
        
        try:
            for i, layer in enumerate(self.target_layers):
                factor = float(scaling_factors[i])
                
                if isinstance(layer, AdaptiveLIFNeuron):
                    orig = layer.base_threshold.data.clone()
                    original_thresholds.append((layer.base_threshold, orig))
                    layer.base_threshold.data.mul_(factor)
                    
                elif isinstance(layer, DualThresholdNeuron):
                    orig_h = layer.threshold_high.data.clone()
                    orig_l = layer.threshold_low.data.clone()
                    original_thresholds.append((layer.threshold_high, orig_h))
                    original_thresholds.append((layer.threshold_low, orig_l))
                    layer.threshold_high.data.mul_(factor)
                    layer.threshold_low.data.mul_(factor)
                    
                elif isinstance(layer, ScaleAndFireNeuron):
                    orig = layer.thresholds.data.clone()
                    original_thresholds.append((layer.thresholds, orig))
                    layer.thresholds.data.mul_(factor)
            
            # 2. 推論と評価
            self.model.eval()
            total_correct = 0
            total_samples = 0
            total_spikes = 0.0
            
            # 高速化のため少数のバッチのみ使用
            max_batches = 5
            for i, batch in enumerate(self.calibration_loader):
                if i >= max_batches: break
                
                # データロード
                inputs: torch.Tensor
                targets: torch.Tensor

                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    key_in = 'input_ids' if 'input_ids' in batch else 'input_images'
                    inputs_raw = batch.get(key_in)
                    targets_raw = batch.get('labels')
                    
                    if inputs_raw is None or targets_raw is None:
                        continue
                    
                    inputs = cast(torch.Tensor, inputs_raw).to(self.device)
                    targets = cast(torch.Tensor, targets_raw).to(self.device)
                else:
                    continue

                # リセット
                if hasattr(self.model, 'reset_spike_stats'):
                    cast(Any, self.model).reset_spike_stats()
                
                # 順伝播
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # 精度計算
                # logits: (Batch, SeqLen, VocabSize) or (Batch, NumClasses)
                preds = logits.argmax(dim=-1)
                
                # --- 修正: ターゲットの形状調整ロジック ---
                # One-hotかどうかは型(float)で判断するのが安全
                if targets.ndim > 1 and targets.shape[-1] > 1 and targets.is_floating_point(): 
                    targets = targets.argmax(dim=-1)
                # ------------------------------------
                
                # シーケンスかスカラーかに関わらずフラット化して比較
                if preds.shape != targets.shape:
                    preds = preds.view(-1)
                    targets = targets.view(-1)
                    
                total_correct += (preds == targets).sum().item()
                total_samples += targets.numel()
                
                # スパイク数取得
                if hasattr(self.model, 'get_total_spikes'):
                    total_spikes += cast(Any, self.model).get_total_spikes()

            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            avg_spikes = total_spikes / total_samples if total_samples > 0 else 0.0
            
            # 3. スコア計算 (最小化対象)
            score = (1.0 - accuracy) * self.metric_weight_accuracy + \
                    (avg_spikes * 0.0001) * self.metric_weight_sparsity
            
            return score

        finally:
            # 4. パラメータの復元
            for param_tensor, orig_data in original_thresholds:
                param_tensor.data.copy_(orig_data)

    def calibrate(self) -> Dict[str, Any]:
        """
        キャリブレーションを実行し、モデルを最適化する。
        """
        logger.info("🚀 Starting Deep Bio-Calibration with HSEO...")
        
        num_params = len(self.target_layers)
        if num_params == 0:
            logger.warning("No target layers found for calibration.")
            return {"status": "skipped"}

        # 探索範囲: 閾値を 0.5倍 〜 2.0倍 の範囲で調整
        bounds = [(0.5, 2.0) for _ in range(num_params)]
        
        # 目的関数のラップ
        def objective_wrapper(particles: np.ndarray) -> np.ndarray:
            scores = []
            for p in particles:
                scores.append(self._evaluate_params(p))
            return np.array(scores)

        # HSEO実行
        best_scales, best_score = optimize_with_hseo(
            objective_function=objective_wrapper,
            dim=num_params,
            num_particles=self.hseo_particles,
            max_iterations=self.hseo_iterations,
            exploration_range=bounds,
            verbose=True
        )
        
        logger.info(f"✅ Calibration Complete. Best Score: {best_score:.4f}")
        
        # 最適パラメータの適用（永続化）
        applied_config = {}
        for i, layer in enumerate(self.target_layers):
            factor = float(best_scales[i])
            layer_name = self.layer_names[i]
            applied_config[layer_name] = factor
            
            if isinstance(layer, AdaptiveLIFNeuron):
                layer.base_threshold.data.mul_(factor)
            elif isinstance(layer, DualThresholdNeuron):
                layer.threshold_high.data.mul_(factor)
                layer.threshold_low.data.mul_(factor)
            elif isinstance(layer, ScaleAndFireNeuron):
                layer.thresholds.data.mul_(factor)
                
        return {
            "best_score": best_score,
            "scaling_factors": applied_config
        }
