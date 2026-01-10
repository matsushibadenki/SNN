# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# 日本語タイトル: Intrinsic Motivation System v2.5 (Phase 2: Intrinsic Reward)
# 目的・内容:
#   ROADMAP Phase 2 "Autonomy" に対応。
#   強化学習のための内発的報酬(Intrinsic Reward)計算メソッドを追加。
#   好奇心(Curiosity)と有能感(Competence)のバランスに基づく自律的な報酬シグナルを生成する。

import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)

# 知識獲得コールバックの型定義
KnowledgeCallback = Callable[[str, str, float, str], None]


class IntrinsicMotivationSystem(nn.Module):
    """
    AIの内発的動機（感情・欲求）を生成するエンジン。
    AsyncBrainKernel (v2.x) と ArtificialBrain (v16.x) の両方に対応。

    Phase 2 Update:
    - calculate_intrinsic_reward(): RLエージェント用の報酬スカラー値を計算
    """

    def __init__(
        self,
        curiosity_weight: float = 1.0,
        boredom_decay: float = 0.995,
        boredom_threshold: float = 0.8,
        novelty_bonus: float = 1.0,  # 新奇性に対する報酬係数
        competence_bonus: float = 0.5  # 課題達成(有能感)に対する報酬係数
    ):
        super().__init__()
        self.curiosity_weight = curiosity_weight
        self.boredom_decay = boredom_decay
        self.boredom_threshold = boredom_threshold
        self.novelty_bonus = novelty_bonus
        self.competence_bonus = competence_bonus

        # 状態履歴（退屈判定用）
        self.last_input_hash: Optional[int] = None
        self.repetition_count = 0

        # 現在の動機状態 (0.0 - 1.0)
        self.drives: Dict[str, float] = {
            "curiosity": 0.5,    # 知的好奇心 (Surpriseに基づく)
            "boredom": 0.0,      # 退屈 (反復に基づく)
            "survival": 1.0,     # 生存本能 (エネルギー残量等)
            "comfort": 0.5,      # 快適さ
            "competence": 0.3    # 有能感 (予測成功やタスク達成に基づく)
        }

        # [Phase 2.1] 知識獲得時のコールバックリスト
        self._knowledge_callbacks: List[KnowledgeCallback] = []

        logger.info(
            "🔥 Intrinsic Motivation System v2.5 (Intrinsic Reward Enabled) initialized.")

    def register_knowledge_callback(self, callback: KnowledgeCallback) -> None:
        """
        知識獲得時に呼び出されるコールバックを登録する。
        CuriosityKnowledgeIntegrator.on_knowledge_acquired を登録することで、
        獲得した知識が自動的に知識グラフへ統合される。
        """
        self._knowledge_callbacks.append(callback)
        logger.debug(
            f"📝 Knowledge callback registered. Total: {len(self._knowledge_callbacks)}")

    def notify_knowledge_acquired(
        self,
        query: str,
        content: str,
        surprise: float,
        source: str = "curiosity_search"
    ) -> None:
        """
        新しい知識を獲得したことを全てのコールバックに通知する。
        """
        for callback in self._knowledge_callbacks:
            try:
                callback(query, content, surprise, source)
            except Exception as e:
                logger.warning(f"⚠️ Knowledge callback error: {e}")

    def process(self, input_payload: Any, prediction_error: Optional[float] = None) -> Dict[str, float]:
        """
        入力に基づいて驚き(Surprise)を計算し、動機状態を更新する。
        """
        surprise = 0.0

        # 1. 予測誤差に基づくSurprise計算
        if prediction_error is not None:
            surprise = min(1.0, prediction_error)

            # 予測誤差による退屈・有能感の更新
            if surprise < 0.1:
                # 予測通り（簡単すぎる） -> 退屈上昇、有能感微増
                self.repetition_count += 1
                boredom_delta = 0.05 * self.repetition_count
                self._update_drive("competence", 0.05)
            else:
                # 驚きがある（未知） -> 退屈解消、好奇心充足
                self.repetition_count = 0
                boredom_delta = -0.2
                # 予測が外れた直後は一時的に有能感が下がるが、学習のチャンス
                self._update_drive("competence", -0.02)

        # 2. ハッシュベースの簡易判定 (予測誤差がない場合)
        elif isinstance(input_payload, (str, int, float)):
            input_hash = hash(input_payload)
            if input_hash == self.last_input_hash:
                self.repetition_count += 1
                surprise = 0.0
                boredom_delta = 0.1 * self.repetition_count
            else:
                self.repetition_count = 0
                surprise = 1.0
                boredom_delta = -0.5
            self.last_input_hash = input_hash
        else:
            boredom_delta = 0.01

        # 値の更新
        self._update_drive("curiosity", surprise * 0.2 - 0.01)  # 自然減衰あり
        self.drives["boredom"] = float(
            np.clip(self.drives["boredom"] + boredom_delta, 0.0, 1.0))

        # ログ出力
        if self.drives["boredom"] > 0.8:
            logger.debug(
                f"🥱 Boredom Level Critical: {self.drives['boredom']:.2f}")
        elif surprise > 0.8:
            logger.debug(
                f"✨ High Surprise ({surprise:.2f})! Curiosity: {self.drives['curiosity']:.2f}")

        return self.get_internal_state()

    def calculate_intrinsic_reward(self, surprise: float, external_reward: float = 0.0) -> float:
        """
        強化学習エージェント用の「内発的報酬」を計算する。
        Reward = 外的報酬 + (好奇心係数 * 新奇性) + (有能感係数 * 有能感) - (退屈ペナルティ)

        Args:
            surprise (float): 観測における予測誤差 (0.0 - 1.0)
            external_reward (float): 環境から得られた外的報酬

        Returns:
            float: 統合された報酬値
        """
        # 新奇性ボーナス (Curiosity Driven)
        # 完全にランダムなノイズ(常にsurprise=1)にハマらないよう、ある程度の予測可能性も重視する「ICM (Intrinsic Curiosity Module)」的アプローチ
        # ここでは簡易的に、現在のCuriosityドライブが高いほど、新しい情報(surprise)に価値を感じるようにする
        novelty_reward = self.novelty_bonus * \
            surprise * self.drives["curiosity"]

        # 退屈ペナルティ
        boredom_penalty = 0.5 * self.drives["boredom"]

        # 有能感ボーナス (Competence)
        # タスクがうまくいっている(Competenceが高い)こと自体を報酬とする
        competence_reward = self.competence_bonus * self.drives["competence"]

        total_reward = external_reward + novelty_reward + \
            competence_reward - boredom_penalty

        return float(total_reward)

    def _update_drive(self, key: str, delta: float):
        """ドライブ値を0.0-1.0の範囲で安全に更新"""
        if key in self.drives:
            self.drives[key] = float(
                np.clip(self.drives[key] + delta, 0.0, 1.0))

    def update_drives(self, surprise: float, energy_level: float, fatigue_level: float, task_success: bool = False) -> Dict[str, float]:
        """ArtificialBrain互換: 環境状態に基づいて全動機を更新"""
        # Curiosity
        if surprise > 0.1:
            self._update_drive("curiosity", 0.1)
        else:
            self._update_drive("curiosity", -0.01)  # 自然減衰

        # Survival (Energy based)
        self.drives["survival"] = max(0.0, 1.0 - (energy_level / 1000.0))

        # Competence
        if task_success:
            self._update_drive("competence", 0.1)
        else:
            self._update_drive("competence", -0.005)  # 失敗または何もしないと自信喪失

        return self.drives

    def get_internal_state(self) -> Dict[str, float]:
        """状態取得 (mypy対応: 値は全てfloatであることを保証)"""
        return {k: float(v) for k, v in self.drives.items()}
