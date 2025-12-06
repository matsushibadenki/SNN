# ファイルパス: snn_research/training/trainers/distillation.py
# Title: Distillation Trainer (知識蒸留) - 戻り値修正版
# Description:
#   BreakthroughTrainerを拡張し、教師モデルからの知識蒸留を行うトレーナー。
#   修正: モデルからの戻り値が3要素を超える場合（SEMMなど）に対応できるよう、
#   アンパック処理を柔軟に変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple, Optional, cast
import logging

from spikingjelly.activation_based import functional # type: ignore

from .breakthrough import BreakthroughTrainer
from snn_research.training.losses import DistillationLoss

logger = logging.getLogger(__name__)

class DistillationTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Union[Tuple[torch.Tensor, ...], Dict[str, Any]], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)
        
        student_input: torch.Tensor
        attention_mask: Optional[torch.Tensor] = None
        student_target: torch.Tensor
        teacher_logits: torch.Tensor

        if isinstance(batch, dict):
            student_input = batch.get('input_ids', batch.get('student_input')).to(self.device) # type: ignore
            student_target = batch.get('labels', batch.get('student_target')).to(self.device) # type: ignore
            teacher_logits = batch.get('teacher_logits').to(self.device) # type: ignore
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            unpacked = [t.to(self.device) for t in batch]
            if len(unpacked) == 4:
                student_input, attention_mask, student_target, teacher_logits = unpacked
            elif len(unpacked) == 3:
                student_input, student_target, teacher_logits = unpacked
                attention_mask = None
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(unpacked)}. Expected 3 or 4.")

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(student_input, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                
                # --- 修正: 柔軟なアンパック処理 ---
                if isinstance(outputs, tuple):
                    # 最低限必要な要素を取得
                    student_logits = outputs[0]
                    spikes = outputs[1] if len(outputs) > 1 else torch.tensor(0.0)
                    mem = outputs[2] if len(outputs) > 2 else torch.tensor(0.0)
                    # 4つ目以降（aux_logitsなど）は蒸留では現在使用しないため無視するか、必要ならここで取得
                else:
                     student_logits, spikes, mem = outputs, torch.tensor(0.0), torch.tensor(0.0)
                # -------------------------------

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
            
            if self.neuron_selector:
                try:
                    switched, reason = self.neuron_selector.step(loss_dict['total'].item())
                    if switched:
                        logger.info(f"NeuronSelector triggered switch: {reason}")
                        self._reinitialize_optimizer()
                except Exception: pass

        # 評価ロジック
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            ignore_idx = self.criterion.ce_loss_fn.ignore_index
            mask = student_target != ignore_idx
            num_valid_tokens = cast(torch.Tensor, mask).sum()
            accuracy = (preds[mask] == student_target[mask]).float().sum() / num_valid_tokens if num_valid_tokens > 0 else torch.tensor(0.0, device=self.device)
            loss_dict['accuracy'] = accuracy
            
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            # config属性への安全なアクセス
            total_time_steps = 16
            if hasattr(model_to_run, 'config'):
                if isinstance(model_to_run.config, dict):
                    total_time_steps = model_to_run.config.get('time_steps', 16)
                else:
                    total_time_steps = getattr(model_to_run.config, 'time_steps', 16)
            elif hasattr(model_to_run, 'time_steps'):
                total_time_steps = getattr(model_to_run, 'time_steps')

            loss_dict['avg_cutoff_steps'] = torch.tensor(float(total_time_steps), device=self.device)
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
