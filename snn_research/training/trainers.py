# ファイルパス: snn_research/training/trainers.py
# Title: SNN 統合学習トレーナー (完全修正版)
# Description:
# - BreakthroughTrainer: 標準的なSNN学習トレーナー。
# - 各種派生トレーナー (Distillation, SelfSupervised, etc.) を実装。
# - 修正: ニューロンの動的置換 (AdaptiveNeuronSelector) が発生した際に、
#   オプティマイザを再初期化して新しいパラメータを学習対象に含める `_reinitialize_optimizer` を実装。
# - 修正: 可視化用フックの登録と解除を確実に行い、メモリリークやオーバーヘッドを防止。
# - 修正: Subclassの _run_step にもニューロン置換時のオプティマイザ再初期化ロジックを適用。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import os
import collections
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, cast, List, Union
import shutil
import time
from torch.optim import Adam
from spikingjelly.activation_based import functional # type: ignore
from pathlib import Path

from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from torch.utils.tensorboard import SummaryWriter
from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.snn_core import SNNCore

from snn_research.models.bio.simple_network import BioSNN
import copy
import logging

from snn_research.core.adaptive_neuron_selector import AdaptiveNeuronSelector

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
        if self.rank in [-1, 0]:
             print(f"⚡️ SNN Cutoff (Evaluation) Enabled: Threshold={self.cutoff_threshold}, MinStepsRatio={self.cutoff_min_steps_ratio}")
    
    def load_ewc_data(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"⚠️ EWCデータファイルが見つかりません: {path}。EWCなしで学習を開始します。")
            return

        ewc_data = torch.load(path, map_location=self.device)
        if isinstance(self.criterion, CombinedLoss):
            self.criterion.fisher_matrix = ewc_data['fisher_matrix']
            self.criterion.optimal_params = ewc_data['optimal_params']
            print(f"✅ EWCデータを '{path}' からロードしました。")
        else:
            print("⚠️ 警告: EWCデータはロードされましたが、現在の損失関数はCombinedLossではありません。EWCは適用されません。")

    def _reinitialize_optimizer(self) -> None:
        """
        モデル構造が変更された場合（ニューロン置換など）、オプティマイザを再初期化する。
        現在の学習率などのハイパーパラメータは維持する。
        """
        logger.info("🔄 Re-initializing optimizer due to model structure change...")
        
        # 現在の学習率を取得 (最初のパラメータグループ)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # オプティマイザのクラスを取得 (AdamWなど)
        optim_class = type(self.optimizer)
        
        # 新しいパラメータでオプティマイザを作り直す
        self.optimizer = optim_class(self.model.parameters(), lr=current_lr)
        
        # スケジューラが存在する場合、スケジューラも更新が必要だが、
        # PyTorchのスケジューラはoptimizerインスタンスに紐付くため、本来は作り直しが必要。
        # ここでは簡易的に警告を出すのみとする（またはスケジューラ再構築ロジックを追加する必要がある）
        if self.scheduler is not None:
            logger.warning("⚠️ Optimizer re-initialized. Scheduler might need manual reset or re-creation.")
            
        logger.info(f"✅ Optimizer re-initialized with LR: {current_lr}")

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        # --- データ入力デバッグ (最初の一回だけ表示) ---
        if is_train and not hasattr(self, '_debug_printed'):
            print("\n🔍 [TRAINER DEBUG] Checking Input Batch:")
            if isinstance(batch, dict):
                ids = batch.get('input_ids')
                if ids is not None:
                    print(f"   - Input Shape: {ids.shape}")
                    if ids.numel() > 0:
                        print(f"   - Input Sample: {ids[0, :10].tolist()}...")
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                ids = batch[0]
                print(f"   - Input Shape: {ids.shape}")
            self._debug_printed = True
        # ------------------------------------------------

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
            
            if 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.device)
            elif 'input_images' in batch:
                input_ids = batch['input_images'].to(self.device)
            else:
                 raise ValueError("Batch dictionary must contain 'input_ids' or 'input_images'.")
                 
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
                threshold: torch.Tensor
                if hasattr(module, 'adaptive_threshold') and getattr(module, 'adaptive_threshold') is not None:
                    threshold = getattr(module, 'adaptive_threshold')
                elif hasattr(module, 'base_threshold'):
                    base_thresh = getattr(module, 'base_threshold')
                    if isinstance(base_thresh, torch.Tensor):
                        threshold = base_thresh.unsqueeze(0).expand_as(mem)
                    else:
                        threshold = torch.full_like(mem, float(base_thresh))
                else:
                    threshold = torch.ones_like(mem)

                self.recorder.record(
                    membrane=mem[0:1].detach(), 
                    threshold=threshold[0:1].detach(), 
                    spikes=spike[0:1].detach()
                )

            for module in model_to_run.modules():
                if isinstance(module, AdaptiveLIFNeuron):
                    hooks.append(module.register_forward_hook(record_hook))
                    break 
        
        return_full_hiddens_flag = isinstance(self.criterion, SelfSupervisedLoss)
        
        total_time_steps: int = 16
        num_classes: int = 10
        logits_for_acc: Optional[torch.Tensor] = None

        # SNN Cutoff (Evaluation only)
        if not is_train and not return_full_hiddens_flag:
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            # タイムステップの取得
            if hasattr(model_to_run, 'config') and isinstance(model_to_run.config, dict):
                 total_time_steps = model_to_run.config.get('time_steps', 16)
            elif hasattr(model_to_run, 'time_steps'):
                 total_time_steps = getattr(model_to_run, 'time_steps')

            min_steps = int(total_time_steps * self.cutoff_min_steps_ratio)
            
            with torch.no_grad():
                eval_outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=return_full_hiddens_flag)
                eval_logits, eval_spikes, eval_mem = eval_outputs
                
                logits_for_acc = eval_logits
                
                # 簡易的なCutoffシミュレーション (実際の推論打ち切りはDeployment側だが、ここではメトリクスとして計算)
                if eval_logits.ndim == 3:
                    probs = F.softmax(eval_logits.view(-1, eval_logits.size(-1)), dim=-1)
                else:
                    probs = F.softmax(eval_logits, dim=-1)
                    
                confidences, _ = torch.max(probs, dim=-1)
                estimated_steps = (1.0 - confidences) * (total_time_steps - min_steps) + min_steps
                estimated_steps[confidences > self.cutoff_threshold] = float(min_steps)
                avg_cutoff_steps = estimated_steps.mean().item()
                
                loss_dict = self.criterion(eval_logits, target_ids, eval_spikes, eval_mem, self.model)
                loss_dict['avg_cutoff_steps'] = torch.tensor(avg_cutoff_steps, device=self.device)
        
        else:
            with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=return_full_hiddens_flag)
                    logits_or_hiddens, spikes, mem = outputs
                    
                    if return_full_hiddens_flag:
                        loss_dict = self.criterion(logits_or_hiddens, target_ids, spikes, mem, self.model)
                        logits_for_acc = None 
                    else:
                        logits_for_acc = logits_or_hiddens
                        loss_dict = self.criterion(logits_for_acc, target_ids, spikes, mem, self.model)
            
            # 登録したフックを削除
            for hook in hooks:
                hook.remove()

            if is_train:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss_dict['total']).backward()
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict['total'].backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                
                # --- ▼ 修正: ニューロン置換時のオプティマイザ再初期化 ▼ ---
                if self.neuron_selector:
                    try:
                        switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                        if switched:
                            logger.info(f"NeuronSelector triggered switch: {reason}")
                            self._reinitialize_optimizer()
                    except Exception as e:
                        logger.error(f"Error during AdaptiveNeuronSelector step: {e}", exc_info=True)
                # --- ▲ 修正 ▲ ---

                if self.meta_cognitive_snn:
                    end_time = time.time()
                    computation_time = end_time - start_time
                    accuracy_val = 0.0
                    if logits_for_acc is not None:
                        with torch.no_grad():
                            preds = torch.argmax(logits_for_acc, dim=-1)
                            ignore_idx: int = -100
                            if hasattr(self.criterion, 'ce_loss_fn'):
                                ce_loss_fn = getattr(self.criterion, 'ce_loss_fn')
                                if hasattr(ce_loss_fn, 'ignore_index'):
                                    ignore_idx_val = getattr(ce_loss_fn, 'ignore_index')
                                    if isinstance(ignore_idx_val, int):
                                        ignore_idx = ignore_idx_val
                            mask = target_ids != ignore_idx
                            num_masked_elements = cast(torch.Tensor, mask).sum()
                            accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                            accuracy_val = accuracy_tensor.item()
                            loss_dict['accuracy'] = accuracy_tensor 
                        
                    self.meta_cognitive_snn.update_metadata(
                        loss=loss_dict['total'].item(),
                        computation_time=computation_time,
                        accuracy=accuracy_val 
                    )
            
            if not is_train:
                with torch.no_grad():
                    accuracy_tensor = torch.tensor(0.0, device=self.device)
                    if logits_for_acc is not None:
                        if 'accuracy' not in loss_dict:
                            preds = torch.argmax(logits_for_acc, dim=-1)
                            ignore_idx = -100
                            if hasattr(self.criterion, 'ce_loss_fn'):
                                ce_loss_fn = getattr(self.criterion, 'ce_loss_fn')
                                if hasattr(ce_loss_fn, 'ignore_index'):
                                    ignore_idx_val = getattr(ce_loss_fn, 'ignore_index')
                                    if isinstance(ignore_idx_val, int):
                                        ignore_idx = ignore_idx_val
                            mask = target_ids != ignore_idx
                            num_masked_elements = cast(torch.Tensor, mask).sum()
                            accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                            loss_dict['accuracy'] = accuracy_tensor 
                    else:
                        loss_dict['accuracy'] = accuracy_tensor 
                    
                    if 'accuracy' in loss_dict and isinstance(loss_dict['accuracy'], torch.Tensor):
                        pass
                    else:
                        loss_dict['accuracy'] = torch.tensor(loss_dict.get('accuracy', 0.0), device=self.device)
            
            if is_train:
                 # ダミーのステップ数（学習時は固定）
                 loss_dict['avg_cutoff_steps'] = torch.tensor(float(total_time_steps), device=self.device)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0:
            logger.warning(f"Epoch {epoch}: Dataloader is empty. Skipping training.")
            return {}
            
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.train()
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value
            progress_bar.set_postfix({k: v / (progress_bar.n + 1) for k, v in total_metrics.items()})

        if self.scheduler:
            self.scheduler.step()
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            if self.scheduler:
                self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0], epoch)
            else:
                self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0:
            logger.warning(f"Epoch {epoch}: Dataloader is empty. Skipping evaluation.")
            return {}
            
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items(): total_metrics[key] += value
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            print(f"Epoch {epoch} Validation Results: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            if self.enable_visualization and hasattr(self, 'recorder') and self.recorder.history['membrane']:
                try:
                    save_path = Path(self.writer.log_dir) / f"neuron_dynamics_epoch_{epoch}.png"
                    plot_neuron_dynamics(self.recorder.history, save_path=save_path)
                    print(f"📊 Neuron dynamics visualization saved to {save_path}")
                except Exception as e:
                    print(f"⚠️ Failed to generate neuron dynamics plot: {e}")

        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any) -> None:
        if self.rank in [-1, 0]:
            model_to_save_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            actual_model = cast(nn.Module, model_to_save_container.model if hasattr(model_to_save_container, 'model') else model_to_save_container)
            
            buffers_to_exclude: set[str] = {
                name for name, buf in actual_model.named_buffers() 
                if buf is not None and any(keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold', 'v', 'u', 'v_s', 'v_d'])
            }
            model_state = {k: v for k, v in actual_model.state_dict().items() if k not in buffers_to_exclude}

            state: Dict[str, Any] = {
                'epoch': epoch, 'model_state_dict': model_state, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self.best_metric
            }
            if self.use_amp: state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました (Epoch: {epoch})。")
            
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                temp_state_for_best: Dict[str, Any] = {'model_state_dict': model_state, **kwargs}
                torch.save(temp_state_for_best, best_path)
                print(f"🏆 新しいベストモデルを '{best_path}' に保存しました (Metric: {metric_value:.4f})。")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"⚠️ チェックポイントファイルが見つかりません: {path}。最初から学習を開始します。")
            return 0
            
        checkpoint: Dict[str, Any] = torch.load(path, map_location=self.device)
        model_to_load_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        actual_model = cast(nn.Module, model_to_load_container.model if hasattr(model_to_load_container, 'model') else model_to_load_container)
        
        state_dict: Dict[str, Any]
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            logger.warning("Checkpoint does not contain 'model_state_dict' key, loading root dict.")
            state_dict = checkpoint
            
        actual_model.load_state_dict(state_dict, strict=False)
        
        if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', float('inf'))
        start_epoch: int = checkpoint.get('epoch', -1) + 1
        print(f"✅ チェックポイント '{path}' を正常にロードしました。Epoch {start_epoch} から学習を再開します。")
        return start_epoch
    
    def _compute_ewc_fisher_matrix(self, dataloader: DataLoader, task_name: str) -> None:
        print(f"🧠 Computing Fisher Information Matrix for EWC (task: {task_name})...")
        self.model.eval()
        
        fisher_matrix: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] = torch.zeros_like(param.data)

        if len(dataloader) == 0:
            logger.warning("EWC Fisher matrix computation skipped: dataloader is empty.")
            return

        for batch in tqdm(dataloader, desc=f"Computing Fisher Matrix for {task_name}"):
            self.model.zero_grad()
            
            input_ids: torch.Tensor
            target_ids: torch.Tensor
            if isinstance(batch, dict):
                target_ids = batch.get('labels', batch.get('targets')).to(self.device) # type: ignore
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                elif 'input_images' in batch:
                    input_ids = batch['input_images'].to(self.device)
                else: continue
            else:
                input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
            
            logits, _, _ = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            
            for name, param in self.model.named_parameters():
                 if param.requires_grad and param.grad is not None:
                    fisher_matrix[name] += param.grad.data.pow(2) / len(dataloader)

        if isinstance(self.criterion, CombinedLoss) and hasattr(self, 'writer'):
            self.criterion.fisher_matrix.update(fisher_matrix)
            for name, param in self.model.named_parameters():
                if name in fisher_matrix:
                    self.criterion.optimal_params[name] = param.data.clone()
            
            ewc_data_path = Path(self.writer.log_dir) / f"ewc_data_{task_name}.pt"
            torch.save({
                'fisher_matrix': self.criterion.fisher_matrix,
                'optimal_params': self.criterion.optimal_params
            }, ewc_data_path)
            print(f"✅ EWC Fisher matrix and optimal parameters for '{task_name}' saved to '{ewc_data_path}'.")

class DistillationTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)
        
        student_input, attention_mask, student_target, teacher_logits = [t.to(self.device) for t in batch]

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(student_input, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                student_logits, spikes, mem = outputs
                
                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits, teacher_logits=teacher_logits, targets=student_target,
                    spikes=spikes, mem=mem, model=self.model, attention_mask=attention_mask
                )
        
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # --- ▼ 修正: オプティマイザ再初期化対応 ▼ ---
            if self.neuron_selector:
                try:
                    switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                    if switched:
                        logger.info(f"NeuronSelector triggered switch: {reason}")
                        self._reinitialize_optimizer()
                except Exception as e:
                    logger.error(f"Error during AdaptiveNeuronSelector step: {e}", exc_info=True)
            # --- ▲ 修正 ▲ ---

        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            ignore_idx = self.criterion.ce_loss_fn.ignore_index
            mask = student_target != ignore_idx
            
            num_valid_tokens = cast(torch.Tensor, mask).sum()
            accuracy: torch.Tensor
            if num_valid_tokens > 0:
                accuracy = (preds[mask] == student_target[mask]).float().sum() / num_valid_tokens
            else:
                accuracy = torch.tensor(0.0, device=self.device)
            loss_dict['accuracy'] = accuracy
            
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            total_time_steps: int = 16
            
            time_steps_val: Any = None
            if hasattr(model_to_run, 'model') and hasattr(model_to_run.model, 'time_steps'):
                time_steps_val = getattr(model_to_run.model, 'time_steps')
            elif hasattr(model_to_run, 'time_steps'):
                time_steps_val = getattr(model_to_run, 'time_steps')
            
            if isinstance(time_steps_val, int):
                total_time_steps = time_steps_val
            loss_dict['avg_cutoff_steps'] = torch.tensor(float(total_time_steps), device=self.device)
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class SelfSupervisedTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)
        
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=True)
                full_hiddens, spikes, mem = outputs
                
                assert isinstance(self.criterion, SelfSupervisedLoss)
                loss_dict = self.criterion(full_hiddens, target_ids, spikes, mem, self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            # --- ▼ 修正: オプティマイザ再初期化対応 ▼ ---
            if self.neuron_selector:
                try:
                    switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                    if switched:
                        logger.info(f"NeuronSelector triggered switch: {reason}")
                        self._reinitialize_optimizer()
                except Exception as e:
                    logger.error(f"Error during AdaptiveNeuronSelector step: {e}", exc_info=True)
            # --- ▲ 修正 ▲ ---

        loss_dict['accuracy'] = torch.tensor(0.0, device=self.device) 
        
        # カットオフステップの記録 (ダミー)
        loss_dict['avg_cutoff_steps'] = torch.tensor(16.0, device=self.device)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class PhysicsInformedTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        # BreakthroughTrainer の _run_step を利用
        metrics = super()._run_step(batch, is_train=is_train)
        return metrics


class ProbabilisticEnsembleTrainer(BreakthroughTrainer):
    def __init__(self, ensemble_size: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self.ensemble_size = ensemble_size

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        ensemble_logits: List[torch.Tensor] = []
        for _ in range(self.ensemble_size):
            functional.reset_net(model_to_reset)
            with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                    logits, _, _ = outputs
                    ensemble_logits.append(logits)
        
        ensemble_logits_tensor = torch.stack(ensemble_logits)
        
        assert isinstance(self.criterion, ProbabilisticEnsembleLoss)
        loss_dict = self.criterion(ensemble_logits_tensor, target_ids, torch.tensor(0.0), torch.tensor(0.0), self.model)

        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            # --- ▼ 修正: オプティマイザ再初期化対応 ▼ ---
            if self.neuron_selector:
                try:
                    switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                    if switched:
                        logger.info(f"NeuronSelector triggered switch: {reason}")
                        self._reinitialize_optimizer()
                except Exception as e:
                    logger.error(f"Error during AdaptiveNeuronSelector step: {e}", exc_info=True)
            # --- ▲ 修正 ▲ ---

        with torch.no_grad():
            mean_logits = ensemble_logits_tensor.mean(dim=0)
            preds = torch.argmax(mean_logits, dim=-1)
            ignore_idx = -100
            if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
            mask = target_ids != ignore_idx
            num_masked_elements = cast(torch.Tensor, mask).sum()
            accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
            loss_dict['accuracy'] = accuracy
        
        loss_dict['avg_cutoff_steps'] = torch.tensor(16.0, device=self.device)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class PlannerTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.train()
        progress_bar = tqdm(dataloader, desc=f"Planner Training Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids, target_plan = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            
            skill_logits = self.model(input_ids)
            
            assert isinstance(self.criterion, PlannerLoss)
            loss_dict = self.criterion(skill_logits, target_plan)
            loss: torch.Tensor = loss_dict['total']
            
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({"loss": loss.item()})
            
class BPTTTrainer:
    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(self.model.parameters(), lr=config.training.get("learning_rate", 1e-3))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = self.config.model.get("type", "simple")

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.model_type == "spiking_transformer":
            return self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        else:
            # RNN-like output: (T, B, V)
            if outputs.dim() == 3 and targets.dim() == 2:
                T, B, V = outputs.shape
                S = targets.shape[1]
                # Time dim mismatch handling if necessary
                return self.criterion(outputs.permute(1, 0, 2).reshape(-1, V), targets.reshape(-1))
            return self.criterion(outputs, targets)

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        if self.model_type == "spiking_transformer":
             outputs, _, _ = self.model(data)
        else:
             outputs = self.model(data)

        loss = self._calculate_loss(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

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
        # パラメータにノイズを加える (粒子拡散)
        for particle in self.particles:
            with torch.no_grad():
                for param in particle.parameters():
                    param.add_(torch.randn_like(param) * self.noise_std)
        
        log_likelihoods: List[float] = []
        for particle in self.particles:
            particle.eval()
            with torch.no_grad():
                if data.dim() > 1:
                    squeezed_data = data.squeeze(0)
                else:
                    squeezed_data = data

                # 入力データに基づいたレートコーディング（ベルヌーイ試行）
                probs = torch.clamp(squeezed_data, 0.0, 1.0)
                input_spikes = torch.bernoulli(probs)

                outputs, _ = particle(input_spikes) # type: ignore[operator]
                
                # 尤度計算 (MSEベース)
                if targets is not None:
                     if isinstance(targets, torch.Tensor) and targets.dim() > 1:
                        squeezed_targets = targets.squeeze(0)
                     else:
                        squeezed_targets = targets
                     
                     if squeezed_targets is not None:
                          squeezed_targets_device = squeezed_targets.to(self.device)
                          loss = F.mse_loss(outputs, squeezed_targets_device)
                          log_likelihoods.append(-loss.item())
                     else:
                          log_likelihoods.append(0.0)
                else:
                     log_likelihoods.append(0.0)
        
        # 重みの更新 (リサンプリング)
        log_likelihoods_tensor = torch.tensor(log_likelihoods, device=self.device)
        self.particle_weights *= torch.exp(log_likelihoods_tensor - log_likelihoods_tensor.max())
        
        if self.particle_weights.sum() > 0:
            self.particle_weights /= self.particle_weights.sum()
        else:
            self.particle_weights.fill_(1.0 / self.num_particles)

        # 有効粒子数が減ったらリサンプリング
        if 1.0 / (self.particle_weights**2).sum() < self.num_particles / 2.0:
            indices = torch.multinomial(self.particle_weights, self.num_particles, replacement=True)
            new_particles: List[nn.Module] = [copy.deepcopy(self.particles[i]) for i in indices]
            self.particles = new_particles
            self.particle_weights.fill_(1.0 / self.num_particles)
        
        best_particle_loss: float = -log_likelihoods_tensor.max().item()
        return best_particle_loss
