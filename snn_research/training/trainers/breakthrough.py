# ファイルパス: snn_research/training/trainers/breakthrough.py
# Title: Breakthrough Trainer (標準SNNトレーナー)
# Description:
#   標準的な代理勾配法によるSNN学習を行うトレーナークラス。
#   他の多くのトレーナーの基底クラスとしても機能する。
#   EWC (Elastic Weight Consolidation) や可視化機能も含む。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import os
import collections
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, cast, List, Union
import time
from torch.optim import Adam
from pathlib import Path
import logging

# 外部ライブラリ
from spikingjelly.activation_based import functional # type: ignore

# プロジェクト内モジュール
from snn_research.training.losses import CombinedLoss, SelfSupervisedLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.adaptive_neuron_selector import AdaptiveNeuronSelector
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class BreakthroughTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool, log_dir: str,
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
        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        
        self.neuron_selector = neuron_selector
        if self.neuron_selector:
            logger.info("✅ AdaptiveNeuronSelector が有効になりました。")

        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging enabled. Log directory: {log_dir}")

        self.enable_visualization = enable_visualization
        if self.enable_visualization and self.rank in [-1, 0]:
            self.recorder = NeuronDynamicsRecorder(max_timesteps=100)
            
        self.cutoff_threshold = cutoff_threshold
        self.cutoff_min_steps_ratio = cutoff_min_steps_ratio
    
    def load_ewc_data(self, path: str) -> None:
        if not os.path.exists(path):
            return
        ewc_data = torch.load(path, map_location=self.device)
        if isinstance(self.criterion, CombinedLoss):
            self.criterion.fisher_matrix = ewc_data['fisher_matrix']
            self.criterion.optimal_params = ewc_data['optimal_params']

    def _reinitialize_optimizer(self) -> None:
        current_lr = self.optimizer.param_groups[0]['lr']
        optim_class = type(self.optimizer)
        self.optimizer = cast(Any, optim_class)(self.model.parameters(), lr=current_lr)

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)

        start_time = time.time()
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids: torch.Tensor
        target_ids: torch.Tensor

        if isinstance(batch, dict):
            target_ids = batch.get('labels')
            if target_ids is None:
                 raise ValueError("Batch dictionary must contain 'labels'.")
            target_ids = target_ids.to(self.device)
            
            # 入力キーの自動判別
            if 'input_images' in batch:
                input_ids = batch['input_images'].to(self.device)
            elif 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.device)
            else:
                 # フォールバック
                 input_ids = list(batch.values())[0].to(self.device)
                 
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if not is_train and self.enable_visualization and self.rank in [-1, 0] and hasattr(self, 'recorder'):
            self.recorder.clear()
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            def record_hook(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                spike, mem = output
                threshold: Optional[torch.Tensor] = None
                if hasattr(module, 'adaptive_threshold') and getattr(module, 'adaptive_threshold') is not None:
                    threshold = getattr(module, 'adaptive_threshold')
                elif hasattr(module, 'base_threshold'):
                    base_thresh = getattr(module, 'base_threshold')
                    if isinstance(base_thresh, torch.Tensor):
                        threshold = base_thresh.unsqueeze(0).expand_as(mem)
                    else:
                        threshold = torch.full_like(mem, float(base_thresh))

                self.recorder.record(
                    membrane=mem[0:1].detach(), 
                    threshold=threshold[0:1].detach() if threshold is not None else None, 
                    spikes=spike[0:1].detach()
                )

            for module in model_to_run.modules():
                if isinstance(module, AdaptiveLIFNeuron):
                    hooks.append(module.register_forward_hook(record_hook))
                    break 
        
        return_full_hiddens_flag = isinstance(self.criterion, SelfSupervisedLoss)
        total_time_steps: int = 16
        logits_for_acc: Optional[torch.Tensor] = None

        # SNN Cutoff (Evaluation only)
        if not is_train and not return_full_hiddens_flag:
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            if hasattr(model_to_run, 'config') and isinstance(model_to_run.config, dict):
                 total_time_steps = model_to_run.config.get('time_steps', 16)
            elif hasattr(model_to_run, 'time_steps'):
                 total_time_steps = getattr(model_to_run, 'time_steps')

            min_steps = int(total_time_steps * self.cutoff_min_steps_ratio)
            
            with torch.no_grad():
                eval_outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=return_full_hiddens_flag)
                
                # アンパック (モデルによってはタプル長が異なる場合があるため注意)
                if isinstance(eval_outputs, tuple):
                    eval_logits = eval_outputs[0]
                    eval_spikes = eval_outputs[1]
                    eval_mem = eval_outputs[2] if len(eval_outputs) > 2 else torch.tensor(0.0)
                else:
                    eval_logits = eval_outputs
                    eval_spikes = torch.tensor(0.0)
                    eval_mem = torch.tensor(0.0)
                
                logits_for_acc = eval_logits
                
                # Cutoff ロジック
                if eval_logits.ndim >= 2:
                    if eval_logits.ndim == 3:
                        probs = F.softmax(eval_logits.view(-1, eval_logits.size(-1)), dim=-1)
                    else:
                        probs = F.softmax(eval_logits, dim=-1)
                        
                    confidences, _ = torch.max(probs, dim=-1)
                    estimated_steps = (1.0 - confidences) * (total_time_steps - min_steps) + min_steps
                    estimated_steps[confidences > self.cutoff_threshold] = float(min_steps)
                    avg_cutoff_steps = estimated_steps.mean().item()
                else:
                    avg_cutoff_steps = float(total_time_steps)
                
                loss_dict = self.criterion(eval_logits, target_ids, eval_spikes, eval_mem, self.model)
                loss_dict['avg_cutoff_steps'] = torch.tensor(avg_cutoff_steps, device=self.device)
        else:
            with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=return_full_hiddens_flag)
                    
                    if isinstance(outputs, tuple):
                        logits_or_hiddens = outputs[0]
                        spikes = outputs[1]
                        mem = outputs[2] if len(outputs) > 2 else torch.tensor(0.0)
                    else:
                        logits_or_hiddens = outputs
                        spikes = torch.tensor(0.0)
                        mem = torch.tensor(0.0)

                    if return_full_hiddens_flag:
                        loss_dict = self.criterion(logits_or_hiddens, target_ids, spikes, mem, self.model)
                        logits_for_acc = None 
                    else:
                        logits_for_acc = logits_or_hiddens
                        loss_dict = self.criterion(logits_for_acc, target_ids, spikes, mem, self.model)
            
            for hook in hooks: hook.remove()

            if is_train:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss_dict['total']).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict['total'].backward()
                    self.optimizer.step()
                
                if self.neuron_selector:
                    try:
                        switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                        if switched:
                            logger.info(f"NeuronSelector triggered switch: {reason}")
                            self._reinitialize_optimizer()
                    except Exception as e:
                        logger.error(f"Error during AdaptiveNeuronSelector step: {e}", exc_info=True)

                if self.meta_cognitive_snn:
                    end_time = time.time()
                    computation_time = end_time - start_time
                    accuracy_val = 0.0
                    if logits_for_acc is not None:
                        with torch.no_grad():
                            preds = torch.argmax(logits_for_acc, dim=-1)
                            ignore_idx = -100
                            if hasattr(self.criterion, 'ce_loss_fn'):
                                ce_loss_fn = getattr(self.criterion, 'ce_loss_fn')
                                if hasattr(ce_loss_fn, 'ignore_index'):
                                    ignore_idx_val = getattr(ce_loss_fn, 'ignore_index')
                                    if isinstance(ignore_idx_val, int): ignore_idx = ignore_idx_val
                            mask = target_ids != ignore_idx
                            num_masked_elements = cast(torch.Tensor, mask).sum()
                            accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                            accuracy_val = accuracy_tensor.item()
                    self.meta_cognitive_snn.update_metadata(loss=loss_dict['total'].item(), computation_time=computation_time, accuracy=accuracy_val)
            
            # --- Accuracy計算の安全化 ---
            accuracy_tensor = torch.tensor(0.0, device=self.device)
            if 'accuracy' not in loss_dict:
                if logits_for_acc is not None:
                    with torch.no_grad():
                        preds = torch.argmax(logits_for_acc, dim=-1)
                        ignore_idx = -100
                        if hasattr(self.criterion, 'ce_loss_fn'):
                            ce_loss_fn = getattr(self.criterion, 'ce_loss_fn')
                            if hasattr(ce_loss_fn, 'ignore_index'):
                                ignore_idx_val = getattr(ce_loss_fn, 'ignore_index')
                                if isinstance(ignore_idx_val, int): ignore_idx = ignore_idx_val
                        mask = target_ids != ignore_idx
                        num_masked_elements = cast(torch.Tensor, mask).sum()
                        accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy_tensor
            
            if is_train:
                 loss_dict['avg_cutoff_steps'] = torch.tensor(float(total_time_steps), device=self.device)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        self.model.train()
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value
            progress_bar.set_postfix({k: v / (progress_bar.n + 1) for k, v in total_metrics.items()})
        if self.scheduler: self.scheduler.step()
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            for key, value in avg_metrics.items(): self.writer.add_scalar(f'Train/{key}', value, epoch)
            lr_val = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/learning_rate', lr_val, epoch)
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items(): total_metrics[key] += value
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            print(f"Epoch {epoch} Validation Results: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items(): self.writer.add_scalar(f'Validation/{key}', value, epoch)
            if self.enable_visualization and hasattr(self, 'recorder') and self.recorder.history['membrane']:
                try:
                    save_path = Path(self.writer.log_dir) / f"neuron_dynamics_epoch_{epoch}.png"
                    plot_neuron_dynamics(self.recorder.history, save_path=save_path)
                except Exception: pass
        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any) -> None:
        if self.rank in [-1, 0]:
            model_to_save_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            actual_model = cast(nn.Module, model_to_save_container.model if hasattr(model_to_save_container, 'model') else model_to_save_container)
            buffers_to_exclude: set[str] = {name for name, buf in actual_model.named_buffers() if buf is not None and any(keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold', 'v', 'u', 'v_s', 'v_d'])}
            model_state = {k: v for k, v in actual_model.state_dict().items() if k not in buffers_to_exclude}
            state: Dict[str, Any] = {'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(), 'best_metric': self.best_metric}
            if self.use_amp: state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                temp_state_for_best: Dict[str, Any] = {'model_state_dict': model_state, **kwargs}
                torch.save(temp_state_for_best, best_path)

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path): return 0
        checkpoint: Dict[str, Any] = torch.load(path, map_location=self.device)
        model_to_load_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        actual_model = cast(nn.Module, model_to_load_container.model if hasattr(model_to_load_container, 'model') else model_to_load_container)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        actual_model.load_state_dict(state_dict, strict=False)
        if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        return checkpoint.get('epoch', -1) + 1
    
    def _compute_ewc_fisher_matrix(self, dataloader: DataLoader, task_name: str) -> None:
        self.model.eval()
        fisher_matrix: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad: fisher_matrix[name] = torch.zeros_like(param.data)
        if len(dataloader) == 0: return
        for batch in tqdm(dataloader, desc=f"Computing Fisher Matrix for {task_name}"):
            self.model.zero_grad()
            input_ids: torch.Tensor
            target_ids: torch.Tensor
            if isinstance(batch, dict):
                target_ids = batch.get('labels', batch.get('targets')).to(self.device) # type: ignore
                input_ids = batch.get('input_ids', batch.get('input_images')).to(self.device) # type: ignore
            else:
                input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
            logits, _, _ = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            for name, param in self.model.named_parameters():
                 if param.requires_grad and param.grad is not None: fisher_matrix[name] += param.grad.data.pow(2) / len(dataloader)
        if isinstance(self.criterion, CombinedLoss) and hasattr(self, 'writer'):
            self.criterion.fisher_matrix.update(fisher_matrix)
            for name, param in self.model.named_parameters():
                if name in fisher_matrix: self.criterion.optimal_params[name] = param.data.clone()
            ewc_data_path = Path(self.writer.log_dir) / f"ewc_data_{task_name}.pt"
            torch.save({'fisher_matrix': self.criterion.fisher_matrix, 'optimal_params': self.criterion.optimal_params}, ewc_data_path)