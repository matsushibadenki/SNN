# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# 日本語タイトル: 前頭前野モジュール v2.2 (Planning Method Added)
# 目的: ArtificialBrainからの呼び出しに対応する plan() メソッドを追加。

from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, TYPE_CHECKING

# 循環インポート防止のため、実行時はインポートせず型チェック時のみ有効化
if TYPE_CHECKING:
    from .global_workspace import GlobalWorkspace
    from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)


class PrefrontalCortex:
    """
    実行制御（Executive Control）を司る前頭前野モジュール。
    ワークスペースを監視し、内発的動機付けに基づいてゴールを再評価する。

    [New Feature] Orthogonal Factorization:
    「脳型AIと直交化の謎」に基づき、ゴール表現と不確実性表現を
    高次元空間内で直交化（直交分解）して保持する。
    これにより、不確実性が高い状況でもゴールの意味内容が干渉を受けず、
    かつ不確実性に応じた動的な柔軟性制御（メタ学習）を実現する。
    """
    # 型アノテーションに文字列を使用し、実行時の依存を排除
    workspace: 'GlobalWorkspace'

    def __init__(
        self,
        workspace: 'GlobalWorkspace',
        motivation_system: 'IntrinsicMotivationSystem',
        d_model: int = 256,   # 高次元ベクトルの次元数
        device: str = 'cpu'
    ):
        """
        Args:
            workspace: GlobalWorkspaceのインスタンス。
            motivation_system: 内発的動機付けシステムのインスタンス。
            d_model: 内部表現ベクトルの次元数。
            device: 計算デバイス。
        """
        self.workspace = workspace
        self.motivation_system = motivation_system
        self.d_model = d_model
        self.device = device

        # --- 既存の状態管理 ---
        self.current_goal: str = "Survive and Explore"
        self.current_context: str = "neutral"
        self.goal_stability: float = 0.0
        self.last_update_reason: str = "initialization"

        # --- [New] 直交化・多重化のための幾何学的状態 ---
        # 不確実性を表現するための固定軸（ランダム初期化後に正規化）
        # 脳内のニューロンポピュレーションにおける「不確実性エンコーディング軸」を模倣
        self.uncertainty_axis = torch.randn(d_model, device=device)
        self.uncertainty_axis = F.normalize(self.uncertainty_axis, p=2, dim=0)

        # 現在のゴールを表すベクトル（初期値はランダムだが、不確実性軸とは直交させる）
        raw_goal = torch.randn(d_model, device=device)
        self.goal_vector = self._project_orthogonally(
            raw_goal, self.uncertainty_axis)

        # 現在の不確実性スカラー（0.0 ~ 1.0）
        self.current_uncertainty_level: float = 0.0

        # ワークスペースのブロードキャストを購読
        if hasattr(self.workspace, 'subscribe'):
            self.workspace.subscribe(self.handle_conscious_broadcast)

        logger.info(
            f"🧠 Prefrontal Cortex (PFC) initialized with Orthogonal Geometry (d={d_model}).")

    def _project_orthogonally(self, target_vec: torch.Tensor, reference_axis: torch.Tensor) -> torch.Tensor:
        """
        [幾何学演算] グラム・シュミットの直交化プロセス。
        target_vec から reference_axis 成分を除去し、純粋な直交成分のみを抽出する。
        これにより、情報の「干渉」を物理的に防ぐ。
        """
        # 射影成分: (v . u) * u
        projection = torch.dot(target_vec, reference_axis) * reference_axis
        orthogonal_vec = target_vec - projection
        return F.normalize(orthogonal_vec, p=2, dim=0)

    def handle_conscious_broadcast(self, source: str, conscious_data: Any) -> None:
        """
        ワークスペースからのブロードキャストを受け取り、エグゼクティブ・コントロールを更新する。
        """
        # 自身が発信源の情報は無視
        if source == "prefrontal_cortex":
            return

        # 動機付けシステムから現在の内部状態を取得
        internal_state = self.motivation_system.get_internal_state()

        # コンテキスト情報の構築
        context = {
            "source": source,
            "content": conscious_data,
            "boredom": internal_state.get("boredom", 0.0),
            "curiosity": internal_state.get("curiosity", 0.0),
            "confidence": internal_state.get("confidence", 0.5)
        }

        self._update_executive_control(context)

    def _update_executive_control(self, context: Dict[str, Any]):
        """
        知覚や感情に基づいて、現在のゴールや行動指針を決定する。
        [Update] 不確実性に基づくメタ制御（安定性と柔軟性のジレンマ解消）を追加。
        """
        source = context["source"]
        content = context["content"]

        # 1. 不確実性の推定とベクトル空間へのマッピング
        # confidence (信頼度) の逆数を不確実性とする
        confidence = context.get("confidence", 0.5)
        self.current_uncertainty_level = 1.0 - confidence

        # 不確実性軸に沿って現在の状態ベクトルを更新（ゴールとは直交しているため干渉しない）
        uncertainty_state_vec = self.uncertainty_axis * self.current_uncertainty_level

        # 2. メタ認知制御：柔軟性（Flexibility）の計算
        # 不確実性が高いほど、ゴール変更に対する抵抗（Inertia）を高める＝安定性重視
        # 不確実性が低い（信頼できる）場合、新しい情報でゴールを即座に更新する＝柔軟性重視
        # ドキュメント「脳型AIと直交化の謎」に基づくロジック
        flexibility_gate = 1.0 - \
            torch.sigmoid(torch.tensor(
                (self.current_uncertainty_level - 0.5) * 5.0)).item()

        new_goal_text: Optional[str] = None
        reason: Optional[str] = None
        salience = 0.5
        force_update = False

        # --- 以下、既存のルールベース決定ロジック ---

        # A. 外部要求（Receptor等）の優先処理
        if source == "receptor" or (isinstance(content, str) and "request" in content.lower()):
            req_text = str(content)
            new_goal_text = f"Fulfill external request: {req_text[:50]}"
            reason = "external_demand"
            salience = 0.9
            force_update = True  # 外部要求は不確実性を無視して割り込む場合がある

        # B. 感情（恐怖・危機）に基づく生存優先
        elif isinstance(content, dict) and content.get("type") == "emotion":
            valence = content.get("valence", 0.0)
            arousal = content.get("arousal", 0.0)
            if valence < -0.7 and arousal > 0.6:
                new_goal_text = "Ensure safety / Avoid negative stimulus"
                reason = "fear_response"
                salience = 1.0
                force_update = True

        # C. 内発的動機（退屈・好奇心）に基づく探索
        elif not new_goal_text:
            if context["boredom"] > 0.8:
                new_goal_text = "Find something new / Explore random"
                reason = "high_boredom"
                salience = 0.7
            elif context["curiosity"] > 0.8:
                topic = getattr(self.motivation_system,
                                'curiosity_context', "unknown")
                new_goal_text = f"Investigate curiosity target: {str(topic)[:30]}"
                reason = "high_curiosity"
                salience = 0.8

        # --- [New] ベクトル幾何学によるゴール更新の調停 ---

        if new_goal_text:
            # 既存ゴールと同じなら無視
            if new_goal_text == self.current_goal:
                return

            # メタ認知ゲートによるフィルタリング
            # force_updateでなければ、不確実性が高いときのゴール変更を抑制する
            if not force_update and flexibility_gate < 0.3:
                logger.info(
                    f"🛡️ PFC Stability Check: Goal update suppressed due to high uncertainty (Flexibility: {flexibility_gate:.2f})")
                return

            safe_reason: str = reason if reason is not None else "context_change"

            logger.info(
                f"🤔 PFC Re-evaluating Goal: '{self.current_goal}' -> '{new_goal_text}' ({safe_reason})")

            # テキスト情報の更新
            self.current_goal = new_goal_text
            self.last_update_reason = safe_reason

            # [New] ゴールベクトルの更新（シミュレーション）
            # 本来はEncoderでテキストを埋め込むが、ここでは新しいランダムベクトルを生成し
            # 不確実性軸と直交化することで「新しい意味」をコードする
            # これにより、不確実性情報（Uncertainty Axis）を破壊せずにゴールだけを書き換える
            proto_goal_vec = torch.randn(self.d_model, device=self.device)
            self.goal_vector = self._project_orthogonally(
                proto_goal_vec, self.uncertainty_axis)

            # 多重化されたPFC全体の状態（Goal + Uncertainty）
            # これは「AI学習・推論における多重化技術調査」にある Task Vector の加算に近い
            pfc_state_vector = self.goal_vector + uncertainty_state_vec

            # ワークスペースへ新しいゴールを提示
            if hasattr(self.workspace, 'upload_to_workspace'):
                self.workspace.upload_to_workspace(
                    source="prefrontal_cortex",
                    data={
                        "type": "goal_setting",
                        "goal": self.current_goal,
                        "reason": safe_reason,
                        "context": self.current_context,
                        "vector_state": pfc_state_vector,  # ベクトル情報も共有可能に
                        "uncertainty": self.current_uncertainty_level
                    },
                    salience=salience
                )

    def plan(self, conscious_content: Any) -> Optional[Dict[str, Any]]:
        """
        [ArtificialBrain Interface]
        現在のゴールと意識の内容（Conscious Content）に基づいて、ハイレベルな行動計画を生成する。
        不確実性が高い場合は「観察」や「探索」を優先する。
        """
        plan_data = {
            "goal": self.current_goal,
            "reason": self.last_update_reason,
            "target": None,
            "directive": "monitor",
            "priority": 0.5
        }

        # 不確実性が高すぎる場合の安全策
        if self.current_uncertainty_level > 0.8:
            plan_data["directive"] = "observe_carefully"
            plan_data["reason"] = "high_uncertainty"
            return plan_data

        # 意識内容に応じたターゲット設定
        if isinstance(conscious_content, dict):
            # 視覚的な特徴があればそれに注目
            if "features" in conscious_content:
                plan_data["target"] = "visual_object"
                plan_data["directive"] = "inspect_visual"
                plan_data["priority"] = 0.8
            # エピソード（記憶の不整合など）があれば
            elif "surprise" in conscious_content:
                plan_data["target"] = "anomaly"
                plan_data["directive"] = "resolve_surprise"
                plan_data["priority"] = 0.9

        elif isinstance(conscious_content, str):
            # 言語的な内容
            plan_data["target"] = "verbal_content"
            plan_data["directive"] = "process_language"

        return plan_data

    def get_executive_context(self) -> Dict[str, Any]:
        """現在のPFCの状態を取得する"""
        return {
            "goal": self.current_goal,
            "context": self.current_context,
            "reason": self.last_update_reason,
            "stability": self.goal_stability,
            # [New] 幾何学的状態の公開
            "uncertainty_level": self.current_uncertainty_level,
            "vector_orthogonality": self._check_orthogonality()  # デバッグ用
        }

    def _check_orthogonality(self) -> float:
        """
        [Debug] 現在のゴールと不確実性軸の直交性を確認する（内積計算）。
        0に近いほど理想的。
        """
        dot_prod = torch.dot(self.goal_vector, self.uncertainty_axis)
        return dot_prod.item()

    def _project_orthogonally_multi(self, target_vec: torch.Tensor, avoidance_axes: list[torch.Tensor]) -> torch.Tensor:
        """
        [高度化] 複数の軸（不確実性、恐怖、ノイズ等）に対して直交化を行う。
        修正グラム・シュミット法 (Modified Gram-Schmidt) を用いて数値的安定性を向上。
        """
        ortho_vec = target_vec.clone()

        for axis in avoidance_axes:
            # 軸自体も正規化されていることを保証
            u = F.normalize(axis, p=2, dim=0)

            # 射影成分を除去
            projection = torch.dot(ortho_vec, u) * u
            ortho_vec = ortho_vec - projection

        return F.normalize(ortho_vec, p=2, dim=0)
