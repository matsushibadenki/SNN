# ファイルパス: snn_research/training/trainers/breakthrough.py
# 日本語タイトル: Breakthrough Trainer (Refactored)
# ファイルの目的・内容:
#   SNN学習のメインループを管理するトレーナー。
#   可読性と保守性を高めるため、巨大なメソッドを分割し、型ヒントを厳密化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Optional, cast, List, Union
import time
import logging
import os
import collections
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional # type: ignore

from snn_research.training.losses import CombinedLoss, SelfSupervisedLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.adaptive_neuron_selector import AdaptiveNeuronSelector

logger = logging.getLogger(__name__)

class BreakthroughTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
        device: str,
        grad_clip_norm: float, 
        rank: int, 
        use_amp: bool, 
        log_dir: str,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        enable_visualization: bool = True,
        cutoff_threshold: float = 0.95,
        cutoff_min_steps_ratio: float = 0.25,
        neuron_selector: Optional[AdaptiveNeuronSelector] = None
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        self.use_amp = use_amp and self.device != 'mps'
        self.log_dir = log_dir
        
        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        self.neuron_selector = neuron_selector
        
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            
        self.enable_visualization = enable_visualization
        if self.enable_visualization and self.rank in [-1, 0]:
            self.recorder = NeuronDynamicsRecorder(max_timesteps=100)
            
        self.cutoff_threshold = cutoff_threshold
        self.cutoff_min_steps_ratio = cutoff_min_steps_ratio

    def _prepare_batch(self, batch: Union[Tuple, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """バッチデータをデバイスに転送し、入力とターゲットに分離する"""
        if isinstance(batch, dict):
            target_ids = batch.get('labels')
            if target_ids is None:
                 raise ValueError("Batch dictionary must contain 'labels'.")
            
            if 'input_images' in batch:
                input_ids = batch['input_images']
            elif 'input_ids' in batch:
                input_ids = batch['input_ids']
            else:
                 input_ids = list(batch.values())[0]
        
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids, target_ids = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
            
        return input_ids.to(self.device), target_ids.to(self.device)

    def _setup_visualization_hooks(self, model: nn.Module) -> List[Any]:
        """可視化用フックを登録する"""
        hooks = []
        if hasattr(self, 'recorder'):
            self.recorder.clear()
            
            def record_hook(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                spike, mem = output
                threshold = getattr(module, 'base_threshold', None)
                if isinstance(threshold, torch.Tensor):
                    threshold = threshold.unsqueeze(0).expand_as(mem)
                elif threshold is not None:
                    threshold = torch.full_like(mem, float(threshold))
                
                self.recorder.record(
                    membrane=mem[0:1].detach(),
                    threshold=threshold[0:1].detach() if threshold is not None else None,
                    spikes=spike[0:1].detach()
                )

            # 最初のAdaptiveLIFNeuronのみ監視
            for module in model.modules():
                if isinstance(module, AdaptiveLIFNeuron):
                    hooks.append(module.register_forward_hook(record_hook))
                    break
        return hooks

    def _forward_pass(self, input_ids: torch.Tensor, is_train: bool) -> Dict[str, Any]:
        """順伝播と推論モードに応じた処理"""
        return_full_hiddens = isinstance(self.criterion, SelfSupervisedLoss)
        
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(
                    input_ids, 
                    return_spikes=True, 
                    return_full_mems=True, 
                    return_full_hiddens=return_full_hiddens
                )
        
        # 出力の正規化 (Tuple -> Dict)
        if isinstance(outputs, tuple):
            result = {
                'logits': outputs[0],
                'spikes': outputs[1],
                'mem': outputs[2] if len(outputs) > 2 else torch.tensor(0.0),
                'aux_logits': outputs[3] if len(outputs) > 3 else None
            }
        else:
            result = {'logits': outputs, 'spikes': torch.tensor(0.0), 'mem': torch.tensor(0.0)}
            
        result['return_full_hiddens'] = return_full_hiddens
        return result

    def _compute_loss(self, forward_result: Dict[str, Any], target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """損失計算"""
        if forward_result['return_full_hiddens']:
            loss_dict = self.criterion(
                forward_result['logits'], # Actually full hiddens
                target_ids, 
                forward_result['spikes'], 
                forward_result['mem'], 
                self.model
            )
        else:
            loss_dict = self.criterion(
                forward_result['logits'], 
                target_ids, 
                forward_result['spikes'], 
                forward_result['mem'], 
                self.model, 
                aux_logits=forward_result.get('aux_logits')
            )
        return loss_dict

    def _backward_pass(self, loss: torch.Tensor):
        """逆伝播と最適化"""
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

    def _run_step(self, batch: Union[Tuple, Dict], is_train: bool) -> Dict[str, Any]:
        """
        1ステップ（1バッチ）の処理フロー。
        """
        # リセット
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)
        
        # モード設定
        self.model.train(is_train)
        
        # データ準備
        input_ids, target_ids = self._prepare_batch(batch)
        
        # 可視化フック
        hooks = []
        if not is_train and self.enable_visualization and self.rank in [-1, 0]:
            hooks = self._setup_visualization_hooks(model_to_reset)

        try:
            start_time = time.time()
            
            # Forward
            forward_result = self._forward_pass(input_ids, is_train)
            
            # Loss
            loss_dict = self._compute_loss(forward_result, target_ids)
            
            # Backward (Train only)
            if is_train:
                self._backward_pass(loss_dict['total'])
                
                # Neuron Selector & Meta-Cognition updates
                if self.neuron_selector:
                     self.neuron_selector.step(loss_dict['total'].item())
                if self.meta_cognitive_snn:
                     acc = 0.0 # Calculate accuracy if needed
                     self.meta_cognitive_snn.update_metadata(loss_dict['total'].item(), time.time()-start_time, acc)

            # Accuracy (Optional, but good for logging)
            if 'accuracy' not in loss_dict and not forward_result['return_full_hiddens']:
                 logits = forward_result['logits']
                 preds = torch.argmax(logits, dim=-1)
                 mask = target_ids != -100 # Ignore padding
                 if mask.sum() > 0:
                     acc = (preds[mask] == target_ids[mask]).float().mean()
                     loss_dict['accuracy'] = acc

        finally:
            for h in hooks: h.remove()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        for batch in pbar:
            metrics = self._run_step(batch, is_train=True)
            for k, v in metrics.items(): total_metrics[k] += v
            pbar.set_postfix({k: v / (pbar.n + 1) for k, v in total_metrics.items()})
            
        if self.scheduler: self.scheduler.step()
        
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            for k, v in avg_metrics.items(): self.writer.add_scalar(f'Train/{k}', v, epoch)
            
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        
        pbar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        with torch.no_grad():
            for batch in pbar:
                metrics = self._run_step(batch, is_train=False)
                for k, v in metrics.items(): total_metrics[k] += v
                
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            print(f"Epoch {epoch} Validation: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for k, v in avg_metrics.items(): self.writer.add_scalar(f'Validation/{k}', v, epoch)
            
            # 可視化画像の保存
            if self.enable_visualization and hasattr(self, 'recorder') and self.recorder.history['membrane']:
                 save_path = Path(self.writer.log_dir) / f"neuron_dynamics_epoch_{epoch}.png"
                 try:
                     plot_neuron_dynamics(self.recorder.history, save_path=save_path)
                 except Exception as e:
                     logger.warning(f"Failed to plot dynamics: {e}")

        return avg_metrics
    
    # save/load_checkpoint 等は省略（変更なし）
    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs):
         # 既存のロジック (省略)
         pass
    
    def load_checkpoint(self, path: str) -> int:
         # 既存のロジック (省略)
         return 0
    
    def load_ewc_data(self, path: str):
         # 既存のロジック (省略)
         pass
