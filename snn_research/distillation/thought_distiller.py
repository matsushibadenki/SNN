# ファイルパス: snn_research/distillation/thought_distiller.py
# 日本語タイトル: Thought Distiller v2.0 (Distribution Matching)
# 目的: 
#   System 2 (Logic/Reasoner) の出力を System 1 (SNN/Intuition) に蒸留する。
#   修正: KLダイバージェンスを用い、教師モデル(System 2)の確信度分布を生徒(System 1)に移転する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

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
        # Teacherの分布をTargetにする
        # T (Temperature) で割ることで分布を平坦化し、教師の「迷い」や「類似クラス情報」も含めて学習する
        
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL(Teacher || Student) = sum(P_teacher * (log P_teacher - log P_student))
        # PyTorchのKLDivLossは input=log_prob, target=prob を期待する
        distillation_loss = F.kl_div(
            soft_prob_student, 
            soft_targets, 
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2) # 勾配のスケールを合わせるため T^2 を掛ける
        
        # 2. Task Loss (Cross Entropy with True Labels)
        student_loss = torch.tensor(0.0, device=student_logits.device)
        if true_labels is not None:
            student_loss = F.cross_entropy(student_logits, true_labels)
        
        # 3. Total Loss
        # alpha * Distill + (1 - alpha) * Task
        if true_labels is None:
            total_loss = distillation_loss
        else:
            total_loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss
            
        self.running_kl_loss = 0.9 * self.running_kl_loss + 0.1 * distillation_loss.detach()
        
        return total_loss

    def get_metrics(self) -> Dict[str, float]:
        return {
            "distillation_kl_loss": self.running_kl_loss.item()
        }