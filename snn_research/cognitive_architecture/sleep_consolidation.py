# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator v3.5 (Phase 2: Autonomous Cycle)
# 目的: HippocampusからCortexへの記憶転送、およびエピソードに基づくGenerative Replayを統括する。

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import random

# 型ヒント用 (循環参照回避のため文字列で指定する場合あり)
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

# ロガー設定
logger = logging.getLogger(__name__)


class SleepConsolidator(nn.Module):
    """
    睡眠時の記憶固定化モジュール。

    Functions:
    1. Memory Transfer: Hippocampus(STM) -> Cortex(LTM)
    2. Generative Replay (Dreaming): 重要な記憶を脳モデル(SNN/VLM)に入力し、
       自己教師あり学習を行うことで重みを調整・定着させる。
    """

    def __init__(
        self,
        memory_system: Any,  # Legacy support
        hippocampus: Optional['Hippocampus'] = None,
        cortex: Optional['Cortex'] = None,
        target_brain_model: Optional[nn.Module] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.memory = memory_system  # Legacy
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.brain_model = target_brain_model

        # 夢の材料となるエピソードバッファ（転送中に一時保持）
        self.dream_seeds: List[str] = []

        self.dream_rate = kwargs.get('dream_rate', 0.1)
        logger.info(
            "🌙 Sleep Consolidator v3.5 (Hippocampus-Cortex Link) initialized.")

    def perform_sleep_cycle(self, duration_cycles: int = 5) -> Dict[str, Any]:
        """
        完全な睡眠サイクルを実行する。
        1. 記憶転送 (Transfer)
        2. 夢によるリプレイ (Replay/Dreaming)
        """
        logger.info(f"🌙 Sleep cycle started for {duration_cycles} cycles.")
        print(f"🌙 Sleep cycle started. Duration: {duration_cycles}")

        # 1. Hippocampus -> Cortex 転送 (Memory Transfer)
        transferred_count = self._transfer_memories()

        loss_history = []
        dreams_replayed = 0

        # 2. Generative Replay (夢を見る)
        if self.brain_model is not None:
            self.brain_model.eval()  # 評価モード（ドロップアウト等を切る）だが、可塑性は有効にする場合がある

            for i in range(duration_cycles):
                # 転送した記憶(dream_seeds)があれば、それを種に夢を見る
                seed_text = random.choice(
                    self.dream_seeds) if self.dream_seeds else None

                clarity = self._dream_step(seed_text=seed_text)
                loss_history.append(clarity)
                dreams_replayed += 1

                if i % 10 == 0:
                    logger.info(
                        f"  ... Dream cycle {i}: Clarity={clarity:.4f}")
        else:
            logger.warning(
                "  Running sleep cycle without brain_model. Skipping dreams.")
            loss_history.extend([0.0 for _ in range(duration_cycles)])

        # 夢の種をクリア（忘却）するか、一部残すかは今後の課題。現在はクリア。
        self.dream_seeds.clear()

        return {
            "consolidated_items": transferred_count,
            "dreams_replayed": dreams_replayed,
            "loss_history": loss_history,
            "status": "COMPLETED"
        }

    def _transfer_memories(self) -> int:
        """
        Hippocampusのバッファから記憶を取り出し、Cortexへ保存する。
        同時に、夢のリプレイ用にローカルバッファ(dream_seeds)にもコピーする。
        """
        if self.hippocampus is None:
            logger.warning("  Hippocampus not connected. Skipping transfer.")
            return 0

        # STMから取り出し（Hippocampusは空になる）
        memories = self.hippocampus.flush_memories()
        count = len(memories)

        if count == 0:
            logger.info("  No new memories to consolidate.")
            return 0

        logger.info(f"  Transferring {count} episodic memories to Cortex...")

        # Cortexへ保存 & 夢の種として保持
        for mem in memories:
            mem_text = str(mem)
            self.dream_seeds.append(mem_text)

            if self.cortex:
                self.cortex.consolidate_episode(mem_text, source="sleep_cycle")

        return count

    def _dream_step(self, seed_text: Optional[str] = None) -> float:
        """
        Generative Replay: 
        直近の記憶(seed_text) または ランダムノイズ から脳活動を生成し、
        Hebbian学習則を通じてネットワークの重みを調整する。
        """
        if self.brain_model is None:
            return 0.0

        try:
            device = next(self.brain_model.parameters()).device

            # 1. 入力生成
            if seed_text:
                # 記憶がある場合: 言語入力をシミュレート (Token ID変換は簡易実装)
                # 本来はTokenizerが必要だが、ここではダミーIDとノイズ画像で代用
                # seed_textの内容によってembeddingを変えるなどの処理が理想
                input_ids = torch.randint(
                    100, 1000, (1, 5), device=device)  # 仮のトークン列
            else:
                # 記憶がない場合: 純粋なランダム夢
                input_ids = torch.tensor(
                    [[101]], device=device, dtype=torch.long)

            # 視覚野への入力: 抽象的な夢 (Gaussian Noise + Pattern)
            noise_image = torch.randn(
                1, 3, 224, 224, device=device) * 0.5 + 0.5

            # 2. 夢を見る (Forward Pass)
            with torch.no_grad():
                # Brain Modelが (input_ids, input_images) を受け取れると仮定
                outputs = self.brain_model(input_ids, input_images=noise_image)

                # 出力の形式に対応 (Tuple or Tensor)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

            # 3. 夢の鮮明度 (Clarity/Confidence)
            # 出力が確率分布(logits)であると仮定
            if isinstance(logits, torch.Tensor):
                probs = F.softmax(logits, dim=-1)
                max_prob, _ = probs.max(dim=-1)
                clarity = max_prob.mean().item()
            else:
                clarity = 0.5  # Default

            # 4. 可塑性更新 (Consolidation Rule)
            # 鮮明な夢(Clarity高) または 記憶に基づく夢(seed_textあり) の場合、結合を強化
            threshold = 0.2 if seed_text else 0.4

            if clarity > threshold:
                self._apply_hebbian_reinforcement(clarity)

            return clarity

        except Exception as e:
            logger.debug(f"Dreaming step skipped/failed: {e}")
            return 0.0

    def _apply_hebbian_reinforcement(self, strength: float):
        """
        単純化されたヘッブ則的強化:
        発火した経路（夢の中で活性化した重み）をわずかに強化する。
        これにより、重要なパターンが「焼き付け」られる。
        """
        if self.brain_model is None:
            return

        # 学習率: 夢の中では非常に小さく設定 (既存知識の破壊を防ぐため)
        reinforcement_factor = 1e-5 * strength

        with torch.no_grad():
            for name, param in self.brain_model.named_parameters():
                if param.requires_grad and "weight" in name:
                    # Hebbian Term: w += alpha * w (Self-reinforcement of active weights)
                    # 厳密なHebb則 (x * y) ではないが、活性化しているパスの重みを増強する簡易実装
                    if param.grad is not None:
                        # Backwardが走っている場合は勾配方向へ
                        param.data -= reinforcement_factor * param.grad
                    else:
                        # Forwardのみの場合は、現在の重み分布を強化（Weight Decayの逆）
                        # ※注意: 発散を防ぐため正規化項が必要だが、ここでは短期的強化のみとする
                        param.data += reinforcement_factor * param.data * 0.01

        logger.debug(
            f"  🧠 Synaptic weights adjusted (Factor: {reinforcement_factor:.2e})")
