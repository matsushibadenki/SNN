# ファイルパス: snn_research/cognitive_architecture/synesthetic_sleep.py
# 日本語タイトル: Synesthetic Sleep Manager (Dream Consolidation) - Type Fixed
# 修正内容: seed_idx の型を int に明示的にキャストし、mypyエラーを解消。

import torch
import torch.nn as nn
import logging
from typing import Dict, List

from snn_research.agent.synesthetic_agent import SynestheticAgent

logger = logging.getLogger(__name__)


class SynestheticSleepManager:
    """
    自律エージェントの睡眠・記憶定着を管理するクラス。

    Process:
    1. REM Sleep (Dreaming): 世界モデルが自律的にシナリオ(夢)を生成。
    2. Consolidation: 生成された夢を「疑似体験」としてBrainに入力し、重みを更新。
    3. Evaluation: 夢の内容とBrainの予測の整合性をチェック。
    """

    def __init__(self, agent: SynestheticAgent, learning_rate: float = 1e-4):
        self.agent = agent
        self.device = agent.device

        # 睡眠学習用のオプティマイザ (Brainのパラメータのみ更新)
        # ※WorldModelは日中に学習済みとする、あるいは睡眠中に整合性を取る
        self.optimizer = torch.optim.AdamW(
            self.agent.brain.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.sleep_history: List[Dict[str, float]] = []

    def enter_sleep_cycle(self, initial_memories: List[Dict[str, torch.Tensor]], num_cycles: int = 5):
        """
        睡眠サイクルを実行する。

        Args:
            initial_memories: 日中に得た「記憶の種」(観測データの断片)のリスト。
            num_cycles: 睡眠の深さ（反復回数）。
        """
        logger.info("🌙 Entering Sleep Mode... (Consolidating Memories)")
        self.agent.brain.train()
        self.agent.world_model.eval()  # 夢を見る側は固定

        total_consolidation_loss = 0.0

        for cycle in range(num_cycles):
            # 1. REM Sleep: Generate Dreams from seeds
            # 記憶の種からランダムに1つ選んで夢を見始める
            # mypy fix: item() returns int|float, explicit cast to int required for indexing
            seed_idx = int(torch.randint(
                0, len(initial_memories), (1,)).item())
            seed_obs = initial_memories[seed_idx]

            # 夢の生成 (Dreaming)
            # horizon: どれくらい先まで夢を見るか
            dream_horizon = 10
            dream_trajectory = self.agent.dream(
                seed_obs, horizon=dream_horizon)

            # 2. Consolidation: Learn from Dreams
            # 夢の中で起きた出来事（感覚入力）に対して、Brainがどう思考するかを学習
            # (自己教師あり学習: 夢の中の文脈整合性を高める)

            loss = self._train_on_dream(dream_trajectory)
            total_consolidation_loss += loss

            if (cycle + 1) % 2 == 0:
                logger.info(
                    f"   💤 REM Cycle {cycle+1}/{num_cycles}: Consolidation Loss = {loss:.4f}")

        avg_loss = total_consolidation_loss / num_cycles
        self.sleep_history.append({'avg_loss': avg_loss, 'cycles': num_cycles})

        logger.info(
            f"🌅 Waking Up... Memory Consolidated (Avg Loss: {avg_loss:.4f})")
        return avg_loss

    def _train_on_dream(self, trajectory: List[Dict[str, torch.Tensor]]) -> float:
        """
        生成された夢の軌道データを用いてBrainを学習させる。
        """
        self.optimizer.zero_grad()

        # 軌道データをバッチ形式に変換 (List[Dict] -> Dict[Tensor])
        # trajectory[t]['vision'] is (B, 1, D)
        # -> combined['vision'] is (B, T, D)
        combined_inputs = {}
        sample_keys = trajectory[0].keys()

        for key in sample_keys:
            # 各ステップのテンソルを結合
            # (B, 1, D) -> list -> (B, T, D)
            tensors = [step[key] for step in trajectory]
            combined_inputs[key] = torch.cat(tensors, dim=1)

        # Brainに入力
        # ここでは「夢の中の感覚入力」を見て、「次の思考(または行動)」を予測させるタスクなどが考えられるが、
        # 簡易的に「Brainの内部状態の安定化（Auto-Associative Learning）」を行う。
        # 具体的には、Brainに通して出力されるLogitsが、何らかのターゲット（ここでは自己回帰的な次ステップ予測など）
        # に合うようにするが、教師データがないため、
        # 「世界モデルの予測」と「Brainの予測」の一致度を高める学習とする。

        # 簡易実装: Brainに夢を見せ、その出力が発散しないように正則化、
        # またはWorldModelが持つ「潜在的な意味」をBrainが言語化(トークン化)できるか試す。

        # ここでは「行動生成の安定化」を目的とし、Brainが出力するActionが極端な値にならないよう学習
        # (本来はもっと複雑な目的関数が必要)

        logits = self.agent.brain(
            text_input=None,  # 言語なしの純粋な夢
            image_input=combined_inputs.get('vision'),
            audio_input=combined_inputs.get('audio'),
            tactile_input=combined_inputs.get('tactile'),
            olfactory_input=combined_inputs.get('olfactory')
        )

        # Loss:
        # 1. 活性のスパース化 (エネルギー効率)
        loss_sparsity = torch.mean(logits ** 2) * 0.01

        # 2. 予測の確信度 (エントロピー最小化) - 夢の中で迷わないように
        # Logits -> Probabilities
        probs = torch.softmax(logits, dim=-1)
        loss_entropy = - \
            torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean()

        total_loss = loss_sparsity + (loss_entropy * 0.1)

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
