# ファイルパス: snn_research/systems/autonomous_learning_loop.py
# 日本語タイトル: Autonomous Learning Loop v2.0 (Phase 2 Integration)
# 目的・内容:
#   ROADMAP Phase 2 "Autonomy" の中核実装。
#   覚醒(Wake)と睡眠(Sleep)のサイクルを管理し、内発的動機に基づく自律学習を行う。
#   EmbodiedVLMAgent, IntrinsicMotivationSystem, SleepConsolidator を統合。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
import logging

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

logger = logging.getLogger(__name__)


class AutonomousLearningLoop:
    """
    自律学習ライフサイクル管理システム (v2.0)

    以下の機能を持つ:
    1. Active Inference: 予測誤差を最小化する行動、または好奇心を最大化する行動の選択。
    2. Intrinsic Reward: 外部報酬がない環境でも「驚き」を報酬として学習。
    3. Homeostasis: エネルギー消費と疲労を管理し、適切なタイミングで睡眠をトリガー。
    4. Sleep Consolidation: 睡眠中に短期記憶をリプレイし、長期記憶へ固定。
    """

    def __init__(
        self,
        agent: EmbodiedVLMAgent,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        energy_capacity: float = 1000.0,
        fatigue_threshold: float = 800.0
    ):
        self.device = device
        self.agent = agent.to(device)
        self.optimizer = optimizer

        # Phase 2 Components
        self.motivator = IntrinsicMotivationSystem().to(device)
        self.sleep_system = SleepConsolidator(agent, optimizer, device=device)

        # World Predictor (予測符号化用ヘッド)
        # エージェントの潜在状態と行動から、次の潜在状態を予測する
        # これにより「予測誤差(Surprise)」を計算可能にする
        fusion_dim = getattr(agent, "fusion_dim", 512)  # デフォルト値またはagentから取得
        action_dim = getattr(agent, "action_dim", 64)

        self.world_predictor = nn.Sequential(
            nn.Linear(fusion_dim + action_dim, 512),
            nn.GELU(),
            nn.Linear(512, fusion_dim)
        ).to(device)

        self.predictor_optimizer = optim.AdamW(
            self.world_predictor.parameters(), lr=1e-3)

        # 恒常性パラメータ (Homeostasis)
        self.energy = energy_capacity
        self.max_energy = energy_capacity
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold

        logger.info("🔄 Autonomous Learning Loop v2.0 (Phase 2) initialized.")

    def step(self,
             current_image: torch.Tensor,
             current_text: torch.Tensor,
             next_image: Optional[torch.Tensor] = None
             ) -> Dict[str, Any]:
        """
        ライフサイクルの1ステップを実行する。

        Returns:
            Dict: 実行結果と現在のステータス
        """
        # 1. 状態チェック (睡眠が必要か？)
        if self._should_sleep():
            return self._perform_sleep_cycle()

        # 覚醒モード (Wake Phase)
        self.agent.train()
        self.world_predictor.train()

        # 2. Agent Perception & Action (SNN Forward)
        # エージェントは現在の観測から行動を決定
        agent_out = self.agent(current_image, current_text)

        z_t = agent_out.get("fused_context")  # 現在の潜在表現 [B, D]
        action = agent_out.get("action_pred")  # 行動ベクトル [B, A]

        # 3. World Prediction (Next State Prediction)
        # 「自分の行動によって世界がどう変わるか」を予測
        # z_{t+1}_pred = P(z_t, action)
        if z_t is not None and action is not None:
            # 次元調整 (Batch次元のみにする)
            if z_t.dim() > 2:
                z_t = z_t.mean(dim=1)

            pred_input = torch.cat([z_t, action], dim=-1)
            z_next_pred = self.world_predictor(pred_input)
        else:
            # 初回などデータ不足時
            z_next_pred = torch.zeros(1, 512).to(self.device)

        # 4. Reality Check (Compute Surprise)
        # 次の時刻の画像が得られている場合（学習時）、予測誤差を計算
        surprise = 0.0
        prediction_loss = torch.tensor(0.0).to(self.device)

        if next_image is not None:
            with torch.no_grad():
                # VLMを使って「実際の」次の潜在表現を取得
                next_out = self.agent.vlm(next_image, current_text)
                z_next_actual = next_out.get("fused_representation")

                if z_next_actual is not None:
                    if z_next_actual.dim() > 2:
                        z_next_actual = z_next_actual.mean(dim=1)

                    # 予測誤差 (MSE) -> これが「驚き」となる
                    prediction_loss = F.mse_loss(z_next_pred, z_next_actual)
                    surprise = torch.clamp(prediction_loss, 0.0, 1.0).item()

        # 5. Intrinsic Motivation & Reward Calculation
        # 内発的動機システムを更新し、報酬を計算
        motivation_state = self.motivator.process(
            input_payload=z_t, prediction_error=surprise)
        intrinsic_reward = self.motivator.calculate_intrinsic_reward(
            surprise=surprise)

        # 6. Memory Storage (Hippocampus)
        # 経験をエピソードとして保存 (睡眠時の学習用)
        # 報酬が高かった(=驚きが大きかった)経験ほど重要
        self.sleep_system.store_experience(
            current_image, current_text, reward=intrinsic_reward)

        # 7. Online Learning (Backprop)
        # 予測モデルとエージェントの更新
        # エージェントは「整合性(Alignment)」と「予測能力」を高めるように学習
        total_loss = prediction_loss + agent_out.get("alignment_loss", 0) * 0.1

        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()
        self.predictor_optimizer.step()

        # 8. Homeostasis Update
        # エネルギー消費と疲労の蓄積
        self.energy -= 1.0  # 活動コスト
        self.fatigue += (0.5 + surprise * 2.0)  # 驚きが大きいほど疲れる

        # 動機状態の更新 (UI表示用)
        drives = self.motivator.update_drives(
            surprise=surprise,
            energy_level=self.energy,
            fatigue_level=self.fatigue,
            task_success=True  # ここでは常に生存中
        )

        return {
            "mode": "wake",
            "step_loss": total_loss.item(),
            "surprise": surprise,
            "intrinsic_reward": intrinsic_reward,
            "energy": self.energy,
            "fatigue": self.fatigue,
            "drives": drives
        }

    def _should_sleep(self) -> bool:
        """睡眠に入るべきか判定"""
        if self.fatigue >= self.fatigue_threshold:
            return True
        if self.energy <= 0:
            return True
        return False

    def _perform_sleep_cycle(self) -> Dict[str, Any]:
        """睡眠サイクルを実行"""
        # 睡眠実行 (Sleep Consolidation)
        sleep_stats = self.sleep_system.sleep(cycles=5)

        # パラメータ回復
        self.fatigue = 0.0
        self.energy = self.max_energy * 0.9  # 完全回復ではない(代謝コスト)

        # 夢ログ
        logger.info(
            f"💤 Slept. Fatigue reset. Loss: {sleep_stats.get('sleep_loss', 0):.4f}")

        return {
            "mode": "sleep",
            "sleep_loss": sleep_stats.get("sleep_loss", 0.0),
            "energy": self.energy,
            "fatigue": self.fatigue,
            "info": "Memory Consolidated"
        }
