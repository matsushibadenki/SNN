# /snn_research/distillation/system_distiller.py
# 日本語タイトル: System 1/2 蒸留マネージャー (System Distiller) v1.0
# 目的・内容: 
#   ReasoningEngine (System 2) の思考結果を SNNCore (System 1) へ蒸留学習させる。
#   - 思考トレースを学習データとして整形。
#   - BitSpikeMamba への知識転移（蒸留）を実行し、推論の「直感化」を促進する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional

# プロジェクト内の依存関係
from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class SystemDistiller:
    """
    System 2 (熟慮) から System 1 (直感) への知能蒸留を管理するクラス。
   
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
        
        # 蒸留用ハイパーパラメータ
        self.temperature = config.get("distill_temperature", 2.0)
        self.alpha = config.get("distill_alpha", 0.5) # 蒸留ロスと通常ロスの比率
        
        # オプティマイザの初期化 (System 1 のみを更新)
        self.optimizer = torch.optim.Adam(
            self.system1.parameters(), 
            lr=config.get("distill_lr", 1e-4)
        )

    async def distill_step(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        1ステップの蒸留プロセス。
        1. System 2 に深く考えさせる。
        2. その結果（教師信号）を System 1 に学習させる。
        """
        # 1. System 2 による「正解」の生成 (Teacher)
        # 蒸留にはエネルギーが必要なため Astrocyte に確認
        if not self.astrocyte.request_resource("distillation_process", 30.0):
            return {"status": "skipped", "reason": "insufficient_energy"}

        # System 2 の熟慮実行
        # 教師信号としてソフトターゲット（logits）や最終回答を取得
        teacher_results = self.system2.process(sensory_input)
        teacher_output = teacher_results.get("final_output")
        
        if teacher_output is None:
            return {"status": "error", "reason": "teacher_no_output"}

        # 2. System 1 による予測 (Student)
        self.system1.train()
        self.optimizer.zero_grad()
        
        student_output = self.system1.forward(sensory_input)
        
        # 3. 蒸留損失の計算
        # System 2 の結論に System 1 を近づける
        if isinstance(student_output, torch.Tensor) and isinstance(teacher_output, torch.Tensor):
            # ソフトターゲット蒸留 (Kullback-Leibler Divergence)
            loss = self._calculate_distill_loss(student_output, teacher_output)
            
            # 4. バックプロパゲーションと更新 (BP依存しない学習則への転換点)
            # 現段階では効率化のため標準的な更新を使用するが、
            # 将来的には平衡伝播法(EP)等に置換可能
            loss.backward()
            self.optimizer.step()
            
            # 統計リセット
            self.system1.reset_state()
            
            return {
                "status": "success",
                "loss": loss.item(),
                "verifier_score": teacher_results.get("verifier_score", 0.0)
            }
            
        return {"status": "skipped", "reason": "incompatible_output_types"}

    def _calculate_distill_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        学生（System 1）と教師（System 2）の出力分布の差異を計算。
        """
        # 出力次元の調整
        if student_logits.shape != teacher_logits.shape:
             # 簡易的な次元合わせ（実際はモデルの vocab_size に依存）
             min_dim = min(student_logits.size(-1), teacher_logits.size(-1))
             student_logits = student_logits[..., :min_dim]
             teacher_logits = teacher_logits[..., :min_dim]

        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

    async def run_consolidation_phase(self, buffer: List[torch.Tensor]):
        """
        睡眠サイクル等で行われる一括蒸留（記憶の固定化）。
       
        """
        logger.info(f"🌙 Starting Consolidation (Distilling {len(buffer)} experiences)...")
        results = []
        for experience in buffer:
            res = await self.distill_step(experience)
            results.append(res)
            await asyncio.sleep(0.01) # 脳内代謝のシミュレート
            
        # 疲労毒素の蓄積を Astrocyte に通知
        self.astrocyte.monitor_neural_activity(firing_rate=len(buffer) * 0.5)
        return results

# ロジックの確認:
# - Objective.md ⑮: 「自信がない時だけ深く考える」結果を蓄積し、次回から直感で動けるようにする基盤。
# - ROADMAP.md Phase 20.1: "System 1/2 Distillation" の項目を具体化。
# - snn_core.py: 蒸留後の状態リセット (reset_state) と連携。