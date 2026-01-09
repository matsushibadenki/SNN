# ファイルパス: snn_research/distillation/system_distiller.py
# 日本語タイトル: System Distiller v1.2 - Async Fix
# 目的: asyncioのインポート漏れ修正と、AstrocyteNetworkの型エラー回避。

import asyncio
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, List

from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class SystemDistiller:
    """
    System 2 (熟慮) の推論プロセスを System 1 (直感) に転移させる蒸留器。
    """

    def __init__(
        self,
        system1: SNNCore,
        system2: ReasoningEngine,
        astrocyte: AstrocyteNetwork,
        config: Dict[str, Any]
    ):
        self.system1 = system1
        self.system2 = system2
        # 型ヒントを明示
        self.astrocyte: AstrocyteNetwork = astrocyte
        self.config = config
        
        self.temperature = float(config.get("distill_temperature", 2.0))
        
        self.optimizer = torch.optim.Adam(
            self.system1.parameters(), 
            lr=float(config.get("distill_lr", 1e-4))
        )

    async def distill_step(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        cost = 30.0
        # [Fix] 明示的なメソッド呼び出し
        if not self.astrocyte.request_resource("distillation_process", cost):
            return {"status": "skipped", "reason": "low_energy"}

        teacher_results = self.system2.process(sensory_input)
        teacher_output = teacher_results.get("final_output")
        
        if teacher_output is None:
            return {"status": "error", "reason": "teacher_no_output"}

        self.system1.train()
        self.optimizer.zero_grad()
        
        student_output = self.system1.forward(sensory_input)
        
        if isinstance(student_output, torch.Tensor) and isinstance(teacher_output, torch.Tensor):
            loss = self._calculate_distill_loss(student_output, teacher_output)
            loss.backward()
            self.optimizer.step()
            self.system1.reset_state()
            
            return {
                "status": "success",
                "loss": loss.item(),
                "verifier_score": teacher_results.get("verifier_score", 0.0)
            }
            
        return {"status": "skipped", "reason": "type_mismatch"}

    def _calculate_distill_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
             min_dim = min(student_logits.size(-1), teacher_logits.size(-1))
             student_logits = student_logits[..., :min_dim]
             teacher_logits = teacher_logits[..., :min_dim]

        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

    async def run_consolidation_phase(self, buffer: List[torch.Tensor]):
        logger.info(f"🌙 Consolidation Phase: Distilling {len(buffer)} experiences...")
        results = []
        for experience in buffer:
            res = await self.distill_step(experience)
            results.append(res)
            await asyncio.sleep(0.01)
            
        # [Fix] 明示的な型キャストまたはメソッド利用
        # AstrocyteNetworkで定義されたメソッドを呼び出す
        rate = float(len(buffer) * 0.5)
        self.astrocyte.monitor_neural_activity(firing_rate=rate)
        return results