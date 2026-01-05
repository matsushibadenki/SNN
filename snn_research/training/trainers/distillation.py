# ファイルパス: snn_research/training/trainers/distillation.py
# Title: Distillation Trainer (Fixed)
# Description:
#   知識蒸留トレーナー。
#   BreakthroughTrainerを継承し、_run_step をオーバーライド。
#   親クラスのメソッド (_reinitialize_optimizer 等) は継承されるため、ここで再定義は不要。

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple, Optional, cast
import logging

from spikingjelly.activation_based import functional  # type: ignore

from .breakthrough import BreakthroughTrainer
from snn_research.training.losses import DistillationLoss

logger = logging.getLogger(__name__)


class DistillationTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Union[Tuple[torch.Tensor, ...], Dict[str, Any]], is_train: bool) -> Dict[str, Any]:
        model_to_reset = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel) else self.model
        functional.reset_net(model_to_reset)

        student_input: torch.Tensor
        attention_mask: Optional[torch.Tensor] = None
        student_target: torch.Tensor
        teacher_logits: torch.Tensor

        if isinstance(batch, dict):
            student_input = batch.get('input_ids', batch.get(
                'student_input')).to(self.device)  # type: ignore
            student_target = batch.get('labels', batch.get(
                'student_target')).to(self.device)  # type: ignore
            teacher_logits_raw = batch.get('teacher_logits')
            if teacher_logits_raw is None:
                raise ValueError(
                    "teacher_logits missing in batch during distillation")
            teacher_logits = teacher_logits_raw.to(
                self.device)  # type: ignore
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
                raise ValueError(
                    f"Unexpected batch tuple length: {len(unpacked)}. Expected 3 or 4.")

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(
                    student_input, return_spikes=True, return_full_mems=True, return_full_hiddens=False)

                student_logits: torch.Tensor
                spikes: torch.Tensor = torch.tensor(0.0)
                mem: torch.Tensor = torch.tensor(0.0)

                if isinstance(outputs, tuple):
                    student_logits = outputs[0]
                    if len(outputs) > 1:
                        spikes = outputs[1]
                    if len(outputs) > 2:
                        mem = outputs[2]
                else:
                    student_logits = outputs

                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits, teacher_logits=teacher_logits, targets=student_target,
                    spikes=spikes, mem=mem, model=self.model, attention_mask=attention_mask
                )

        if is_train:
            if self.optimizer is None:
                raise ValueError("Optimizer is not defined")
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.neuron_selector:
                try:
                    switched, reason = self.neuron_selector.step(
                        loss_dict['total'].item())
                    if switched:
                        logger.info(
                            f"NeuronSelector triggered switch: {reason}")
                        # 親クラスのメソッドを呼び出す
                        self._reinitialize_optimizer()
                except Exception:
                    pass

        # 精度計算 (リファクタリング: 親クラスのヘルパーを使うと綺麗だが、ここでは独立して実装)
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            ignore_idx = self.criterion.ce_loss_fn.ignore_index
            mask = student_target != ignore_idx
            num_valid_tokens = cast(torch.Tensor, mask).sum()

            if num_valid_tokens > 0:
                acc_tensor = (preds[mask] == student_target[mask]
                              ).float().sum() / num_valid_tokens
            else:
                acc_tensor = torch.tensor(0.0, device=self.device)

            # Dict[str, Any] なので Tensor を代入してもOK
            loss_dict['accuracy'] = acc_tensor

            # ... (cutoff steps calculation code if needed) ...
            loss_dict['avg_cutoff_steps'] = torch.tensor(
                16.0, device=self.device)  # Dummy or calculate

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
