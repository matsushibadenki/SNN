# ファイルパス: snn_research/training/trainers/breakthrough.py
# 日本語タイトル: Breakthrough Trainer (Graph Fix v4)
# 修正: RuntimeError (Trying to backward through the graph a second time) を防ぐため、
#       SNNCoreのreset_stateを明示的に呼び出し、状態の完全なデタッチを保証。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Optional, cast, List, Union

import logging
import os
import collections
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pathlib
from spikingjelly.activation_based import functional  # type: ignore

from snn_research.training.losses import CombinedLoss, SelfSupervisedLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.adaptive_neuron_selector import AdaptiveNeuronSelector
from snn_research.training.base_trainer import AbstractTrainer

logger = logging.getLogger(__name__)


class BreakthroughTrainer(AbstractTrainer):
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
        neuron_selector: Optional[AdaptiveNeuronSelector] = None,
        config: Optional[Any] = None  # Added config arg
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=os.path.join(log_dir, "checkpoints") if log_dir else None,
            config=config
        )
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        # Fix: Ensure AMP is only enabled if CUDA is available, as GradScaler in PyTorch < 2.0 (or typical usage) often warns/fails on CPU/MPS
        # Even with torch.amp.autocast, GradScaler usually requires CUDA.
        self.use_amp = use_amp and torch.cuda.is_available()
        self.log_dir = log_dir

        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        self.neuron_selector = neuron_selector

        # self.device, self.model, self.optimizer are handled by super

        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        # best_metric is initialized in super

        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)

        self.enable_visualization = enable_visualization
        if self.enable_visualization and self.rank in [-1, 0]:
            self.recorder = NeuronDynamicsRecorder(max_timesteps=100)

        self.cutoff_threshold = cutoff_threshold
        self.cutoff_min_steps_ratio = cutoff_min_steps_ratio

    def load_ewc_data(self, path: str) -> None:
        """EWC用のFisher行列をロードする"""
        if not os.path.exists(path):
            return
        ewc_data = torch.load(path, map_location=self.device)
        if isinstance(self.criterion, CombinedLoss):
            self.criterion.fisher_matrix = ewc_data['fisher_matrix']
            self.criterion.optimal_params = ewc_data['optimal_params']

    def _reinitialize_optimizer(self) -> None:
        """オプティマイザを再初期化する（ニューロン置換時などに使用）"""
        if self.optimizer is None:
            logger.warning("Optimizer is not defined, cannot reinitialize.")
            return
        current_lr = self.optimizer.param_groups[0]['lr']
        optim_class = type(self.optimizer)
        self.optimizer = cast(Any, optim_class)(
            self.model.parameters(), lr=current_lr)

    def _compute_ewc_fisher_matrix(self, dataloader: DataLoader, task_name: str) -> None:
        """EWC用のFisher行列を計算し保存する"""
        self.model.eval()
        fisher_matrix: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] = torch.zeros_like(param.data)

        if len(dataloader) == 0:
            return

        for batch in tqdm(dataloader, desc=f"Computing Fisher Matrix for {task_name}"):
            self.model.zero_grad()
            input_ids, target_ids = self._prepare_batch(batch)

            outputs = self.model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_matrix[name] += param.grad.data.pow(
                        2) / len(dataloader)

        if isinstance(self.criterion, CombinedLoss) and hasattr(self, 'writer'):
            self.criterion.fisher_matrix.update(fisher_matrix)
            self.criterion.optimal_params = {}
            for name, param in self.model.named_parameters():
                if name in fisher_matrix:
                    self.criterion.optimal_params[name] = param.data.clone()

            ewc_data_path = pathlib.Path(
                self.writer.log_dir) / f"ewc_data_{task_name}.pt"
            torch.save({
                'fisher_matrix': self.criterion.fisher_matrix,
                'optimal_params': self.criterion.optimal_params
            }, ewc_data_path)

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

    def _calculate_accuracy(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """精度を計算するヘルパーメソッド"""
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            ignore_idx = -100

            if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index  # type: ignore

            mask = target_ids != ignore_idx
            num_valid = mask.sum()

            if num_valid > 0:
                acc_val = (preds[mask] == target_ids[mask]
                           ).float().sum() / num_valid
                return acc_val
            else:
                return torch.tensor(0.0, device=self.device)

    def _run_step(self, batch: Union[Tuple, Dict], is_train: bool) -> Dict[str, Any]:
        """1ステップの処理を実行"""
        model_to_reset = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel) else self.model

        # [Fix] Graph残留を防ぐための強力なリセット
        # 1. SpikingJelly標準のリセット
        functional.reset_net(model_to_reset)

        # 2. SNNCore等のラッパーが持つカスタムリセットの呼び出し
        if hasattr(model_to_reset, 'reset_state') and callable(model_to_reset.reset_state):
            model_to_reset.reset_state()

        # 3. その他、resetメソッドを持つサブモジュールの強制リセット
        for module in model_to_reset.modules():
            if hasattr(module, 'reset') and callable(module.reset):
                module.reset()

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = self._prepare_batch(batch)

        hooks: List[Any] = []

        if not is_train and self.enable_visualization and self.rank in [-1, 0] and hasattr(self, 'recorder'):
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
                    threshold=threshold[0:1].detach(
                    ) if threshold is not None else None,
                    spikes=spike[0:1].detach()
                )

            for module in self.model.modules():
                if isinstance(module, AdaptiveLIFNeuron):
                    hooks.append(module.register_forward_hook(record_hook))
                    break

        try:

            return_full_hiddens_flag = isinstance(
                self.criterion, SelfSupervisedLoss)

            # AMP (Automatic Mixed Precision)
            device_type = self.device.type if self.device.type != 'mps' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(
                        input_ids,
                        return_spikes=True,
                        return_full_mems=True,
                        return_full_hiddens=return_full_hiddens_flag
                    )

                    logits: Optional[torch.Tensor] = None
                    spikes = torch.tensor(0.0)
                    mem = torch.tensor(0.0)
                    aux_logits = None

                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        if len(outputs) > 1:
                            spikes = outputs[1]
                        if len(outputs) > 2:
                            mem = outputs[2]
                        if len(outputs) > 3:
                            aux_logits = outputs[3]
                    else:
                        logits = outputs

                    if return_full_hiddens_flag:
                        loss_dict = self.criterion(
                            logits, target_ids, spikes, mem, self.model)
                        logits_for_acc = None
                    else:
                        logits_for_acc = logits
                        loss_dict = self.criterion(
                            logits, target_ids, spikes, mem, self.model, aux_logits=aux_logits)

            if is_train:
                if self.optimizer is None:
                    raise ValueError("Optimizer is not defined")
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss_dict['total']).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict['total'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                if self.neuron_selector:
                    self.neuron_selector.step(loss_dict['total'].item())

            final_loss_dict: Dict[str, Any] = {
                k: v for k, v in loss_dict.items()}

            if 'accuracy' not in final_loss_dict:
                if logits_for_acc is not None:
                    acc_tensor = self._calculate_accuracy(
                        logits_for_acc, target_ids)
                    final_loss_dict['accuracy'] = acc_tensor
                else:
                    final_loss_dict['accuracy'] = torch.tensor(
                        0.0, device=self.device)

            # [Fix] 戻り値はプリミティブ型にしてグラフを切断する
            return {k: v.item() if torch.is_tensor(v) else v for k, v in final_loss_dict.items()}

        finally:
            for h in hooks:
                h.remove()

    # type: ignore[override]
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(
            float)  # type: ignore
        num_batches = len(dataloader)
        if num_batches == 0:
            return {}

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(
            self.rank not in [-1, 0]))

        for batch in pbar:
            metrics = self._run_step(batch, is_train=True)
            for k, v in metrics.items():
                total_metrics[k] += v
            pbar.set_postfix({k: v / (pbar.n + 1)
                             for k, v in total_metrics.items()})

        if self.scheduler:
            self.scheduler.step()

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f'Train/{k}', v, epoch)

        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        if num_batches == 0:
            return {}

        pbar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(
            self.rank not in [-1, 0]))
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items():
                    total_metrics[key] += value

        avg_metrics = {key: value / num_batches for key,
                       value in total_metrics.items()}

        if self.rank in [-1, 0] and hasattr(self, 'writer'):
            print(f"Epoch {epoch} Validation Results: " +
                  ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)

            if self.enable_visualization and hasattr(self, 'recorder') and self.recorder.history['membrane']:
                try:
                    save_path = pathlib.Path(
                        self.writer.log_dir) / f"neuron_dynamics_epoch_{epoch}.png"
                    plot_neuron_dynamics(
                        self.recorder.history, save_path=save_path)
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")

        return avg_metrics

    # type: ignore[override]
    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any) -> None:  # type: ignore
        if self.rank in [-1, 0]:
            model_to_save_container = self.model.module if isinstance(
                self.model, nn.parallel.DistributedDataParallel) else self.model
            actual_model = cast(nn.Module, model_to_save_container.model if hasattr(
                model_to_save_container, 'model') else model_to_save_container)
            buffers_to_exclude = {name for name, buf in actual_model.named_buffers() if buf is not None and any(
                keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold', 'v', 'u', 'v_s', 'v_d'])}
            model_state = {k: v for k, v in actual_model.state_dict(
            ).items() if k not in buffers_to_exclude}

            if self.optimizer is None:
                raise ValueError(
                    "Optimizer must be set before saving checkpoint")

            state: Dict[str, Any] = {
                'epoch': epoch, 'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(), 'best_metric': self.best_metric
            }
            if self.use_amp:
                state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)

            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(
                    os.path.dirname(path), 'best_model.pth')
                temp_state_for_best = {
                    'model_state_dict': model_state, **kwargs}
                torch.save(temp_state_for_best, best_path)

    def load_checkpoint(self, path: str) -> int:  # type: ignore[override]
        if not os.path.exists(path):
            return 0
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load_container = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel) else self.model
        actual_model = cast(nn.Module, model_to_load_container.model if hasattr(
            model_to_load_container, 'model') else model_to_load_container)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        actual_model.load_state_dict(state_dict, strict=False)

        if 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                # オプティマイザがまだ初期化されていない場合は初期化を試みる、あるいはwarning
                logger.warning(
                    "Optimizer is not initialized, skipping optimizer state load.")
            else:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        return int(checkpoint.get('epoch', -1)) + 1
