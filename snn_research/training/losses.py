# ファイルパス: snn_research/training/losses.py
# Title: SNN 損失関数定義 (Fix: PhysicsInformedLoss kwargs support)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, cast
from transformers import PreTrainedTokenizerBase

from snn_research.core.layers.complex_attention import MultiLevelSpikeDrivenSelfAttention


class CombinedLoss(nn.Module):
    """
    クロスエントロピー損失、各種正則化、EWC損失、およびMoE負荷分散損失を組み合わせた統合損失関数。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        ce_weight: float = 1.0,
        spike_reg_weight: float = 0.0,
        mem_reg_weight: float = 0.0,
        sparsity_reg_weight: float = 0.0,
        temporal_compression_weight: float = 0.0,
        sparsity_threshold_reg_weight: float = 0.0,
        target_spike_rate: float = 0.02,
        ewc_weight: float = 0.0,
        moe_load_balancing_weight: float = 0.0,
        **kwargs
    ):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight,
            'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight,
            'sparsity_threshold_reg': sparsity_threshold_reg_weight,
            'ewc': ewc_weight,
            'moe_load_balancing': moe_load_balancing_weight
        }
        self.target_spike_rate = target_spike_rate
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        spikes: torch.Tensor,
        mem: torch.Tensor,
        model: nn.Module,
        aux_logits: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        ce_loss = self.ce_loss_fn(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        # モデル側で正規化済みと仮定
        spike_rate = spikes.mean()

        # --- 競合の解消 ---
        if self.weights['spike_reg'] > 0:
            spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(
                self.target_spike_rate, device=spike_rate.device))
            sparsity_loss = torch.tensor(0.0, device=logits.device)
        else:
            spike_reg_loss = torch.tensor(0.0, device=logits.device)
            sparsity_loss = spike_rate

        mem_reg_loss = torch.mean(mem**2)

        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(
                0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        sparsity_threshold_reg_loss = torch.tensor(0.0, device=logits.device)
        if self.weights['sparsity_threshold_reg'] > 0:
            threshold_sum = torch.tensor(0.0, device=logits.device)
            count = 0
            for module in model.modules():
                if isinstance(module, MultiLevelSpikeDrivenSelfAttention):
                    threshold_sum += module.sparsity_threshold
                    count += 1
            if count > 0:
                sparsity_threshold_reg_loss = - (threshold_sum / count)

        ewc_loss = torch.tensor(0.0, device=logits.device)
        if self.weights['ewc'] > 0 and self.fisher_matrix:
            for name, param in model.named_parameters():
                if name in self.fisher_matrix and param.requires_grad:
                    fisher = self.fisher_matrix[name].to(param.device)
                    opt_param = self.optimal_params[name].to(param.device)
                    ewc_loss += (fisher * (param - opt_param)**2).sum()

        moe_loss = torch.tensor(0.0, device=logits.device)
        if self.weights.get('moe_load_balancing', 0.0) > 0 and aux_logits is not None:
            probs = F.softmax(aux_logits, dim=-1)
            expert_usage = probs.mean(dim=[0, 1, 2])
            num_experts = aux_logits.shape[-1]
            target_usage = torch.full_like(expert_usage, 1.0 / num_experts)
            moe_loss = F.mse_loss(expert_usage, target_usage) * num_experts

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss +
                      self.weights['sparsity_threshold_reg'] * sparsity_threshold_reg_loss +
                      self.weights['ewc'] * ewc_loss +
                      self.weights.get('moe_load_balancing', 0.0) * moe_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate,
            'temporal_compression_loss': temporal_compression_loss,
            'sparsity_threshold_reg_loss': sparsity_threshold_reg_loss,
            'ewc_loss': ewc_loss,
            'moe_loss': moe_loss
        }


class DistillationLoss(nn.Module):
    """
    知識蒸留のための損失関数（各種正則化付き）。

    [Phase 2.4 Update] 温度スケジューリングとスパイク発火率正則化強化:
    - 動的温度スケジューリング: 学習初期は高温(soft)、後期は低温(hard)
    - スパイク発火率正則化の強化: 目標発火率の維持を厳格化
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 0.3, distill_weight: float = 0.7,
                 spike_reg_weight: float = 0.01, mem_reg_weight: float = 0.0, sparsity_reg_weight: float = 0.00001,
                 temporal_compression_weight: float = 0.0, sparsity_threshold_reg_weight: float = 0.0,
                 temperature: float = 2.0, target_spike_rate: float = 0.02,
                 # [Phase 2.4] 新パラメータ
                 temperature_schedule: str = "constant",  # "constant", "linear", "cosine"
                 initial_temperature: float = 4.0,
                 final_temperature: float = 1.0,
                 spike_rate_tolerance: float = 0.01,  # 発火率の許容範囲
                 **kwargs):
        super().__init__()
        student_pad_id = tokenizer.pad_token_id
        self.temperature = temperature
        self.weights = {
            'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight, 'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight,
            'sparsity_threshold_reg': sparsity_threshold_reg_weight
        }
        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=student_pad_id if student_pad_id is not None else -100)
        self.distill_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)
        self.target_spike_rate = target_spike_rate

        # [Phase 2.4] 温度スケジューリング
        self.temperature_schedule = temperature_schedule
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.spike_rate_tolerance = spike_rate_tolerance
        self._current_epoch = 0
        self._total_epochs = 1

    def set_epoch_info(self, current_epoch: int, total_epochs: int) -> None:
        """
        温度スケジューリング用のエポック情報を設定する。
        Trainerから各エポック開始時に呼び出す。
        """
        self._current_epoch = current_epoch
        self._total_epochs = max(1, total_epochs)

    def get_current_temperature(self) -> float:
        """
        スケジュールに基づいて現在の温度を計算する。
        """
        if self.temperature_schedule == "constant":
            return self.temperature

        progress = self._current_epoch / self._total_epochs

        if self.temperature_schedule == "linear":
            # 線形減衰: 初期→最終へ線形に変化
            return self.initial_temperature + (self.final_temperature - self.initial_temperature) * progress

        elif self.temperature_schedule == "cosine":
            # コサイン減衰: より滑らかな遷移
            import math
            return self.final_temperature + 0.5 * (self.initial_temperature - self.final_temperature) * (1 + math.cos(math.pi * progress))

        return self.temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:

        # [Phase 2.4] 動的温度を取得
        current_temp = self.get_current_temperature()

        is_classification = student_logits.ndim == 2

        if is_classification:
            ce_loss = self.ce_loss_fn(student_logits, targets)
        else:
            ce_loss = self.ce_loss_fn(
                student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 動的温度を使用したソフトラベル
        soft_student_log_probs = F.log_softmax(
            student_logits / current_temp, dim=-1)
        soft_teacher_log_probs = F.log_softmax(
            teacher_logits / current_temp, dim=-1)
        distill_loss_unreduced = self.distill_loss_fn(
            soft_student_log_probs, soft_teacher_log_probs).sum(dim=-1)

        num_valid_tokens: torch.Tensor
        if is_classification:
            mask = torch.ones(
                targets.shape[0], dtype=torch.bool, device=targets.device)
            num_valid_tokens = torch.tensor(
                targets.shape[0], device=targets.device)
            masked_distill_loss = distill_loss_unreduced
        else:
            if attention_mask is None:
                mask = (targets != self.ce_loss_fn.ignore_index)
            else:
                mask = attention_mask.bool()
            num_valid_tokens = cast(torch.Tensor, mask).sum()
            masked_distill_loss = distill_loss_unreduced.where(
                mask, torch.tensor(0.0, device=distill_loss_unreduced.device))

        if num_valid_tokens.item() > 0:
            distill_loss = masked_distill_loss.sum() / num_valid_tokens.item()
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)

        distill_loss = distill_loss * (current_temp ** 2)

        spike_rate = spikes.mean()

        if self.weights['spike_reg'] > 0:
            target_spike_rate = torch.tensor(
                self.target_spike_rate, device=spikes.device)
            spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)
            sparsity_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            spike_reg_loss = torch.tensor(0.0, device=student_logits.device)
            sparsity_loss = spike_rate

        mem_reg_loss = torch.mean(mem**2)

        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(
                0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        sparsity_threshold_reg_loss = torch.tensor(
            0.0, device=student_logits.device)
        if self.weights['sparsity_threshold_reg'] > 0:
            threshold_sum = torch.tensor(0.0, device=student_logits.device)
            count = 0
            for module in model.modules():
                if isinstance(module, MultiLevelSpikeDrivenSelfAttention):
                    threshold_sum += module.sparsity_threshold
                    count += 1
            if count > 0:
                sparsity_threshold_reg_loss = - (threshold_sum / count)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss +
                      self.weights['sparsity_threshold_reg'] * sparsity_threshold_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss,
            'sparsity_loss': sparsity_loss, 'mem_reg_loss': mem_reg_loss,
            'spike_rate': spike_rate,
            'temporal_compression_loss': temporal_compression_loss,
            'sparsity_threshold_reg_loss': sparsity_threshold_reg_loss
        }


class SelfSupervisedLoss(nn.Module):
    def __init__(self, prediction_weight: float, spike_reg_weight: float, mem_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02, tcl_weight: float = 1.0, tcl_temperature: float = 0.1, **kwargs):
        super().__init__()
        self.weights = {
            'prediction': prediction_weight,
            'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight,
            'tcl': tcl_weight
        }
        self.tcl_temperature = tcl_temperature
        self.target_spike_rate = target_spike_rate
        pad_id = tokenizer.pad_token_id
        self.prediction_loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -100)

    def forward(self, full_hiddens: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        B, S, T, D = full_hiddens.shape
        hiddens_flat = full_hiddens.permute(0, 1, 2, 3).reshape(B * S * T, D)
        hiddens_norm = F.normalize(hiddens_flat, p=2, dim=1)
        hiddens_st = full_hiddens.reshape(B * S, T, D)
        anchors = hiddens_st[:, :-1, :].reshape(-1, D)
        anchors = hiddens_st[:, :-1, :].reshape(-1, D)

        ignore_index = self.prediction_loss_fn.ignore_index
        targets_expanded_time = targets.unsqueeze(2).repeat(1, 1, T-1)
        valid_mask = (targets_expanded_time != ignore_index).reshape(-1)

        similarity_matrix = torch.matmul(
            anchors, hiddens_norm.T) / self.tcl_temperature
        positive_indices = torch.arange(anchors.size(0), device=anchors.device)
        tcl_loss_unmasked = F.cross_entropy(
            similarity_matrix, positive_indices, reduction='none')

        if valid_mask.numel() != tcl_loss_unmasked.numel():
            tcl_loss = tcl_loss_unmasked.mean()
        else:
            num_valid = valid_mask.sum().clamp(min=1)
            tcl_loss = (tcl_loss_unmasked *
                        valid_mask.float()).sum() / num_valid

        spike_rate = spikes.mean()

        if self.weights['spike_reg'] > 0:
            target_spike_rate = torch.tensor(
                self.target_spike_rate, device=spikes.device)
            spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)
        else:
            spike_reg_loss = torch.tensor(0.0, device=spikes.device)

        mem_reg_loss = torch.mean(mem**2)

        total_loss = (self.weights['tcl'] * tcl_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_reg'] * mem_reg_loss)

        return {
            'total': total_loss,
            'tcl_loss': tcl_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_reg_loss': mem_reg_loss,
            'spike_rate': spike_rate
        }


class PhysicsInformedLoss(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 1.0, spike_reg_weight: float = 0.0, mem_smoothness_weight: float = 0.0, target_spike_rate: float = 0.02, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_smoothness': mem_smoothness_weight,
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem_sequence: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        """
        Args:
            **kwargs: aux_logits などを吸収するために追加
        """
        ce_loss = self.ce_loss_fn(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        spike_rate = spikes.mean()

        if self.weights['spike_reg'] > 0:
            spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(
                self.target_spike_rate, device=spike_rate.device))
        else:
            spike_reg_loss = torch.tensor(0.0, device=spike_rate.device)

        mem_smoothness_loss = torch.tensor(0.0, device=logits.device)
        if isinstance(mem_sequence, torch.Tensor) and mem_sequence.numel() > 1 and mem_sequence.ndim > 0:
            time_dim = -2 if mem_sequence.ndim >= 3 else 0
            if mem_sequence.shape[time_dim] > 1:
                try:
                    mem_diff = torch.diff(mem_sequence, dim=time_dim)
                    mem_smoothness_loss = torch.mean(mem_diff**2)
                except RuntimeError:
                    pass

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_smoothness'] * mem_smoothness_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_smoothness_loss': mem_smoothness_loss,
            'spike_rate': spike_rate
        }


class PlannerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, predicted_logits: torch.Tensor, target_plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target_plan.view(-1)
        loss = self.loss_fn(predicted_logits, target)
        return {'total': loss, 'planner_loss': loss}


class ProbabilisticEnsembleLoss(nn.Module):
    def __init__(self, ce_weight: float, variance_reg_weight: float, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {'ce': ce_weight, 'variance_reg': variance_reg_weight}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        mean_logits = logits.mean(dim=0)
        ce_loss = self.ce_loss_fn(
            mean_logits.view(-1, mean_logits.size(-1)), targets.view(-1))

        probs = F.softmax(logits, dim=-1)
        variance = probs.var(dim=0).mean()
        variance_reg_loss = variance

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['variance_reg'] * variance_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'variance_reg_loss': variance_reg_loss
        }
