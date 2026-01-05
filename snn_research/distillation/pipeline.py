# ファイルパス: snn_research/distillation/pipeline.py
# 日本語タイトル: Advanced Distillation Pipeline (Specialist & CoT)
# 目的・内容:
#   ROADMAP v16.5 "Distillation and T=1 Integration" および Phase 5-H の実装。
#   1. Specialist Distillation: 複数の専門家モデルを単一の生徒モデルに統合。
#   2. CoT Distillation: ReasoningEngineの思考プロセス(System 2)を
#      軽量モデル(System 1)に蒸留し、直感的な推論能力を向上させる。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """
    オンメモリまたは生成された蒸留用データセット。
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_distillation_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    蒸留用バッチ処理。入力ID、注意マスク、教師の出力（Logits/Thoughts）をまとめる。
    """
    # 簡易実装: すべての要素がTensorであることを前提にstack/padする
    # 実際にはpadding処理が必要
    input_ids = [item['input_ids'] for item in batch]
    labels = [item.get('labels', item['input_ids']) for item in batch]

    # Padding (簡易的)
    max_len = max(x.size(0) for x in input_ids)
    padded_inputs = torch.stack(
        [F.pad(x, (0, max_len - x.size(0))) for x in input_ids])
    padded_labels = torch.stack(
        [F.pad(x, (0, max_len - x.size(0))) for x in labels])

    batch_out = {
        'input_ids': padded_inputs,
        'labels': padded_labels
    }

    if 'teacher_logits' in batch[0]:
        teacher_logits = [item['teacher_logits'] for item in batch]
        padded_logits = torch.stack(
            [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in teacher_logits])
        batch_out['teacher_logits'] = padded_logits

    if 'thought_trace' in batch[0]:
        # テキストデータ等はリストのまま渡すか、別途エンコードが必要
        pass

    return batch_out


class AdvancedDistillationPipeline:
    """
    高度な知識蒸留パイプライン。
    Specialist統合とCoT蒸留をサポートする。
    """

    def __init__(
        self,
        student_model: nn.Module,
        device: str = "cpu",
        tokenizer: Any = None
    ):
        self.student_model = student_model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        logger.info(
            f"⚗️ AdvancedDistillationPipeline initialized on {device}.")

    def run_specialist_distillation(
        self,
        specialist_models: List[nn.Module],
        unlabeled_dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-4,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        [Multi-Teacher Distillation]
        複数の専門家モデル（例: 画像認識、言語処理、音声解析）の出力を統合し、
        単一の生徒モデル（汎用SNN）に学習させる。

        Args:
            specialist_models: 教師モデルのリスト。各データに対してアンサンブルまたは担当分けで推論する。
            unlabeled_dataset: ラベルなしデータセット（蒸留用）。
            alpha: 蒸留損失とタスク損失のバランス係数。
        """
        logger.info(
            f"👨‍🏫 Starting Specialist Distillation with {len(specialist_models)} teachers...")

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        dataloader = DataLoader(
            unlabeled_dataset, batch_size=batch_size, shuffle=True)

        # 教師モデルを評価モードに
        for model in specialist_models:
            model.to(self.device).eval()

        self.student_model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                # データの準備 (Datasetの実装依存だが、ここではinput_idsを想定)
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                else:
                    inputs = batch.to(self.device)  # Tuple等の場合

                # 1. 教師のアンサンブル推論
                teacher_logits_sum = None
                with torch.no_grad():
                    for teacher in specialist_models:
                        # 各教師の推論 (APIが統一されている前提)
                        # SFormer/ReasoningEngine互換の forward -> (logits, ...)
                        out = teacher(inputs)
                        logits = out[0] if isinstance(out, tuple) else out

                        if teacher_logits_sum is None:
                            teacher_logits_sum = logits
                        else:
                            teacher_logits_sum += logits

                    if teacher_logits_sum is None:
                        # 教師モデルが存在しない、または出力が得られなかった場合
                        logger.warning(
                            "No teacher logits computed. Skipping batch.")
                        continue

                    # 平均化 (Soft Target)
                    teacher_avg_logits = teacher_logits_sum / \
                        len(specialist_models)  # type: ignore

                # 2. 生徒の推論
                student_out = self.student_model(inputs)
                student_logits = student_out[0] if isinstance(
                    student_out, tuple) else student_out

                # 3. 損失計算 (KL Divergence)
                loss_distill = self._compute_distillation_loss(
                    student_logits, teacher_avg_logits, temperature
                )

                # (Optional) Hard Label Loss (もしラベルがあれば)
                loss = loss_distill  # ここでは蒸留損失のみ

                # 4. 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

            logger.info(
                f"✅ Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")

    def run_cot_distillation(
        self,
        reasoning_engine: ReasoningEngine,
        prompts: List[str],
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 1e-5
    ):
        """
        [Chain-of-Thought Distillation]
        System 2 (ReasoningEngine) が生成した「思考過程と結論」を教師データとして生成し、
        System 1 (生徒モデル) に学習させる。これにより、生徒は「直感的に正しい推論」ができるようになる。

        Process:
        1. Promptを入力。
        2. ReasoningEngineが思考(Thought)と回答(Answer)を生成。
        3. 生徒モデルに `Prompt -> Thought -> Answer` のシーケンスを学習させる。
        """
        logger.info(
            f"🧠 Starting CoT Distillation (System 2 -> System 1) with {len(prompts)} prompts...")

        if not self.tokenizer:
            raise ValueError("Tokenizer is required for CoT distillation.")

        # 1. データセット生成 (Rollout)
        logger.info("   Generating reasoning traces...")
        cot_data = []

        for prompt in tqdm(prompts, desc="Reasoning Rollout"):
            # Tokenize
            input_ids = self.tokenizer.encode(
                prompt, return_tensors='pt').to(self.device)

            # System 2 Reasoning
            # ReasoningEngine.think_and_solve は検証済みのベストな回答を返す
            result = reasoning_engine.think_and_solve(
                input_ids, task_type="general")

            if result['final_output'] is not None:
                # 思考過程と思考結果を結合したテキストを作成
                # thought_trace はリストなので結合

                # ReasoningEngineの実装次第だが、final_outputはID列
                answer_ids = result['final_output']

                # ここでは簡易的に、Teacherが出力したID列をそのまま正解ラベルとする
                # (Prompt + Generated IDs)
                full_ids = torch.cat([input_ids, answer_ids], dim=1)

                cot_data.append({
                    'input_ids': full_ids.squeeze(0),  # (SeqLen)
                    # Causal LM training (next token prediction)
                    'labels': full_ids.squeeze(0)
                })

        if not cot_data:
            logger.warning("⚠️ No valid CoT data generated. Aborting.")
            return

        dataset = DistillationDataset(cot_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_distillation_batch
        )

        # 2. 蒸留トレーニング (Fine-tuning)
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        self.student_model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(
                dataloader, desc=f"CoT Distill Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 生徒モデルの学習
                # 生徒はSFormer等のBaseModelを想定 (Auto-regressive loss内蔵または自前計算)
                out = self.student_model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out

                # Shift for causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else -100
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        logger.info(
            "✅ CoT Distillation complete. System 1 has absorbed System 2's reasoning patterns.")

    def _compute_distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        Soft Targetに対するKL Divergence損失を計算する。
        """
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)

        # KLDivLoss: reduction='batchmean' is recommended
        distill_loss = F.kl_div(soft_prob, soft_targets,
                                reduction='batchmean') * (temperature ** 2)
        return distill_loss

    def evaluate_student(self, test_dataset: Dataset) -> Dict[str, float]:
        """生徒モデルの性能評価"""
        self.student_model.eval()
        # 簡易評価ロジック（実際にはAccuracyなどを計算）
        return {"accuracy": 0.0}  # Dummy
