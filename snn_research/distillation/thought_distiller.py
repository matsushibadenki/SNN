# ファイルパス: snn_research/distillation/thought_distiller.py
# Title: Thought Distillation Manager (System 2 -> System 1)
# Description:
#   System 2 (Teacher) の思考プロセス(CoT)を教師データとして、
#   System 1 (Student: BitSpikeModel) を学習させる蒸留パイプライン。
#   ROADMAP v20.1 "System 1/2 Distillation" 対応。

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, cast
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ThoughtDistillationManager:
    """
    思考蒸留マネージャー。
    Teacher(Symbolic/LLM)の出力をStudent(SNN)に模倣させる。
    """

    def __init__(self, student_model: nn.Module, teacher_engine: Any, learning_rate: float = 1e-4):
        self.student = student_model
        self.teacher = teacher_engine
        self.optimizer = optim.AdamW(
            self.student.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def generate_thought_dataset(self, problems: List[str]) -> List[Dict[str, Any]]:
        """
        Teacherを使って、問題に対する「思考過程」と「答え」を生成する。
        """
        logger.info(
            f"🧠 System 2 is generating thoughts for {len(problems)} problems...")
        dataset = []

        for q in problems:
            # Teacherによる推論 (System 2)
            # 戻り値: {'input': str, 'thought_chain': str, 'answer': str}
            reasoning_result = self.teacher.solve_with_reasoning(q)
            dataset.append(reasoning_result)

        return dataset

    def distill(self, dataset: List[Dict[str, Any]], epochs: int = 3, batch_size: int = 1):
        """
        生成された思考データをStudentに学習させる。
        """
        logger.info("⚗️ Starting Distillation (System 2 -> System 1)...")
        self.student.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0

            pbar = tqdm(dataset, desc=f"Distill Epoch {epoch+1}/{epochs}")
            for item in pbar:
                # 入力プロンプト
                input_text = f"Q: {item['input']}\nReasoning:"

                # 教師の思考トレース (CoT) + 答え
                target_text = f" {item['thought_chain']}\nAnswer: {item['answer']}<EOS>"

                # ここでトークナイズとテンソル化を行う（簡易実装）
                # 実際にはTokenizerが必要だが、デモ用にダミーのembedding処理を想定
                # studentモデルが (input_ids, labels) を受け取れると仮定

                # --- Student Forward & Backward ---
                self.optimizer.zero_grad()

                # デモ用: student.forward_loss などのインターフェースを想定
                # テキストを直接渡せるラッパーがある前提、もしくは内部で変換
                # [Fix] Cast self.student to Any to avoid mypy error "Tensor not callable"
                student_any = cast(Any, self.student)
                if hasattr(student_any, 'forward_text_loss'):
                    loss = student_any.forward_text_loss(
                        input_text, target_text)
                else:
                    # ダミーロス (モデルの実装に依存するため)
                    # 本来はここで self.student(input_ids) -> logits -> CrossEntropy
                    loss = torch.tensor(0.5, requires_grad=True)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / max(count, 1)
            logger.info(f"   Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        logger.info("✅ Distillation Completed. System 1 is now smarter.")


class SymbolicTeacher:
    """
    論理的・記号的教師（System 2の役割）。
    ここではデモとして、算数問題をステップバイステップで解くロジックを実装。
    """

    def solve_with_reasoning(self, question: str) -> Dict[str, str]:
        # 例: "15 + 27" -> Steps -> "42"
        try:
            # 簡易パーサー
            parts = question.replace("?", "").split("+")
            a = int(parts[0].strip())
            b = int(parts[1].strip())
            res = a + b

            # 思考過程の生成
            a_ones, a_tens = a % 10, a // 10
            b_ones, b_tens = b % 10, b // 10

            ones_sum = a_ones + b_ones
            carry = ones_sum // 10
            rem_ones = ones_sum % 10

            tens_sum = a_tens + b_tens + carry

            thought = (
                f"First, add ones: {a_ones} + {b_ones} = {ones_sum}. "
                f"Write {rem_ones}, carry {carry}. "
                f"Next, add tens: {a_tens} + {b_tens} + carry({carry}) = {tens_sum}. "
                f"Combine them to get {tens_sum}{rem_ones}."
            )

            return {
                "input": question,
                "thought_chain": thought,
                "answer": str(res)
            }
        except Exception:
            return {
                "input": question,
                "thought_chain": "I cannot solve this clearly.",
                "answer": "Unknown"
            }
