# /snn_research/distillation/system_distiller.py
# 日本語タイトル: System 1/2 蒸留マネージャー (System Distiller) v1.1
# 目的・内容: 
#   ReasoningEngine (System 2) の思考結果を SNNCore (System 1) へ蒸留学習させる。
#   - 思考の「直感化」を促進するための、非同期蒸留ループの実装。
#   - mypy エラー (asyncio 未定義) を修正。

import asyncio  # 修正: インポートを追加
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional

# プロジェクト内の既存コンポーネント
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
        self.astrocyte = astrocyte
        self.config = config
        
        self.temperature = config.get("distill_temperature", 2.0)
        
        # System 1 (BitSpikeMamba等) の重みを更新対象とする
        self.optimizer = torch.optim.Adam(
            self.system1.parameters(), 
            lr=config.get("distill_lr", 1e-4)
        )

    async def distill_step(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        1ステップの蒸留。System 2 の知見を教師信号にする。
        """
        # 1. リソース確認
        cost = 30.0
        if not self.astrocyte.request_resource("distillation_process", cost):
            return {"status": "skipped", "reason": "low_energy"}

        # 2. System 2 (Teacher) の熟慮実行
        teacher_results = self.system2.process(sensory_input)
        teacher_output = teacher_results.get("final_output")
        
        if teacher_output is None:
            return {"status": "error", "reason": "teacher_no_output"}

        # 3. System 1 (Student) の学習
        self.system1.train()
        self.optimizer.zero_grad()
        
        student_output = self.system1.forward(sensory_input)
        
        if isinstance(student_output, torch.Tensor) and isinstance(teacher_output, torch.Tensor):
            # 蒸留損失 (KL Divergence) の計算
            loss = self._calculate_distill_loss(student_output, teacher_output)
            
            # 誤差逆伝播 (将来的に生物学的学習則への移行を想定)
            loss.backward()
            self.optimizer.step()
            
            # モデル状態のリセット
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
        """
        教師モデルと学生モデルの出力分布の差異を計算。
        """
        # 必要に応じて次元を合わせる
        if student_logits.shape != teacher_logits.shape:
             min_dim = min(student_logits.size(-1), teacher_logits.size(-1))
             student_logits = student_logits[..., :min_dim]
             teacher_logits = teacher_logits[..., :min_dim]

        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

    async def run_consolidation_phase(self, buffer: List[torch.Tensor]):
        """
        一括蒸留（記憶の固定化）。
        mypy修正: asyncio.sleep を使用可能に。
        """
        logger.info(f"🌙 Consolidation Phase: Distilling {len(buffer)} experiences...")
        results = []
        for experience in buffer:
            res = await self.distill_step(experience)
            results.append(res)
            # 非同期スリープによるオーバーヘッド抑制
            await asyncio.sleep(0.01) 
            
        # 代謝監視の更新
        self.astrocyte.monitor_neural_activity(firing_rate=len(buffer) * 0.5)
        return results