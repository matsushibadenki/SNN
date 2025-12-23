# ファイルパス: snn_research/distillation/thought_distiller.py
# 日本語タイトル: Thought Distiller v2.1 (Type Fixed)
# 目的: 
#   System 2 (Logic/Reasoner) の出力を System 1 (SNN/Intuition) に蒸留する。
#   修正: running_kl_loss の型エラー解消。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, cast

class ThoughtDistiller(nn.Module):
    """
    Thought Distillation Module.
    Logic -> Intuition Transfer.
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature # 蒸留温度 (ソフトターゲットの滑らかさ)
        self.alpha = alpha             # 蒸留ロスとタスクロスの重み付け
        
        # 評価用メトリクス
        self.register_buffer('running_kl_loss', torch.tensor(0.0))

    def forward(self, 
                student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor, 
                true_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            student_logits: SNNの出力 (System 1)
            teacher_logits: Reasoning Engineの出力 (System 2)
            true_labels: 正解ラベル (Optional)
        
        Returns:
            Total Loss
        """
        # 1. Distillation Loss (KL Divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            soft_prob_student, 
            soft_targets, 
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2) 
        
        # 2. Task Loss (Cross Entropy with True Labels)
        student_loss = torch.tensor(0.0, device=student_logits.device)
        if true_labels is not None:
            student_loss = F.cross_entropy(student_logits, true_labels)
        
        # 3. Total Loss
        if true_labels is None:
            total_loss = distillation_loss
        else:
            total_loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss
            
        # Update metric with explicit type casting/handling
        # running_kl_loss is a buffer (Tensor), so update in-place or reassignment
        current_val = self.running_kl_loss # type: ignore
        new_val = 0.9 * current_val + 0.1 * distillation_loss.detach()
        self.running_kl_loss = new_val # type: ignore
        
        return total_loss

    def get_metrics(self) -> Dict[str, float]:
        # cast to Tensor to satisfy mypy if needed, though runtime is fine
        val = self.running_kl_loss.item() # type: ignore
        return {
            "distillation_kl_loss": val
        }
