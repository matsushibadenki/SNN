# ファイルパス: snn_research/agent/synesthetic_agent.py
# 日本語タイトル: Synesthetic Autonomous Agent (Full Implementation)
# 目的: Brain v4 (思考) と World Model (予測) を統合した自律エージェント。
#       dream() メソッドにより、行動前の脳内シミュレーションを実現する。

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import logging

from snn_research.models.experimental.brain_v4 import SynestheticBrain
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)


class SynestheticAgent(BaseModel):
    """
    五感統合エージェント。

    Architecture:
    1. Perception: 五感入力 -> SynestheticBrain & WorldModel
    2. World Model: 直近の観測から「現在の状態」を推定し、「未来」を予測する。
    3. Brain (Policy): 世界モデルの状態と言語的思考(Context)を組み合わせて、次の行動(Action)を決定する。
    """

    def __init__(
        self,
        brain: SynestheticBrain,
        world_model: SpikingWorldModel,
        action_dim: int,
        device: str = 'cpu'
    ):
        super().__init__()
        self.brain = brain
        self.world_model = world_model
        self.action_dim = action_dim
        self.device = device

        # 行動生成用ヘッド (Brainの出力を行動に変換)
        # Brainの出力次元(vocab_size or d_model) -> Action Dim
        # 簡易的にBrainのd_modelを入力とする
        self.actor_head = nn.Linear(brain.d_model, action_dim)

        # 短期記憶 (Short-term Memory) / 思考のコンテキスト
        self.thought_context: Optional[torch.Tensor] = None

        self.to(device)

    def reset(self):
        """エージェントの状態リセット"""
        self.thought_context = None
        # 必要に応じて内部ステートのリセット処理を追加

    def step(
        self,
        observations: Dict[str, torch.Tensor],
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        """
        1ステップの行動決定ループ (Perceive -> Think -> Act)

        Args:
            observations: 現在の観測 {'vision': ..., 'tactile': ...}
            instruction: 外部からの言語指示 (Optional)
        Returns:
            action: 実行すべき行動ベクトル (B, ActionDim)
        """
        # 観測が空の場合はエラー
        if not observations:
            raise ValueError("Observations cannot be empty.")

        # 1. World Modelによる状態推定 (State Estimation)
        with torch.no_grad():
            # WorldModelのEncoderを利用して潜在状態 z_t を取得
            encoded_obs = {}
            for mod, data in observations.items():
                try:
                    # BrainのEncoderを利用 (SynestheticBrainはUniversalSpikeEncoderを持つ)
                    encoded_obs[mod] = self.brain.encoder.encode(
                        data, modality=mod)
                except ValueError:
                    # サポートされていないモダリティはスキップ
                    continue

            if not encoded_obs:
                raise ValueError(
                    "No valid modalities encoded from observations.")

            # 世界モデル視点のコンテキスト (直感的把握)
            _ = self.world_model.projector(
                encoded_obs)  # (B, T, D)

        # 2. Brainによる思考と意思決定 (Thinking & Decision Making)
        # 言語指示がある場合はトークン化
        text_input = None
        if instruction:
            # 簡易トークナイズ
            text_input = self.brain.encoder.encode_text_str(instruction)
            if text_input.dim() == 2:
                text_input = text_input.unsqueeze(0)
            text_input = text_input.to(self.device)

        # Brain Forward
        # Brain v4.forwardはlogitsを返す
        logits = self.brain(
            text_input=text_input,
            image_input=observations.get('vision'),
            audio_input=observations.get('audio'),
            tactile_input=observations.get('tactile'),
            olfactory_input=observations.get('olfactory')
        )

        # 3. Action Generation
        # Brainの思考結果(logits)から特徴を抽出して行動を決定
        features = torch.mean(logits, dim=1)  # (B, Vocab)

        # 次元圧縮 (Vocab -> D_model)
        # 注意: 本来は__init__で定義すべき層だが、Brainの内部仕様と疎結合にするためここで動的に定義・適用する簡易実装
        # (実運用ではBrainの出力層をActorCritic形式にするのが望ましい)
        feature_projector = nn.Linear(
            features.shape[-1], self.brain.d_model, device=self.device)
        # 学習時は勾配が必要だが、step()は推論のみを想定しているケースが多い。
        # ここでは推論として扱う。
        with torch.no_grad():
            projected_features = feature_projector(features)
            action = torch.tanh(self.actor_head(
                projected_features))  # -1 ~ 1 の連続値行動

        return action

    def dream(self, initial_obs: Dict[str, torch.Tensor], horizon: int = 10) -> List[Dict[str, torch.Tensor]]:
        """
        世界モデルを用いたシミュレーション（夢）。
        現在の状況から、「もし何もしなかったら（あるいはランダムに動いたら）どうなるか」を予測する。

        Args:
            initial_obs: 現在の観測
            horizon: 予測する未来のステップ数
        Returns:
            trajectory: 予測された観測のリスト [{'vision': ..., 'tactile': ...}, ...]
        """
        self.world_model.eval()
        trajectory = []
        current_obs = initial_obs

        # バッチサイズの取得
        batch_size = next(iter(initial_obs.values())).shape[0]

        with torch.no_grad():
            for t in range(horizon):
                # 1. 行動の決定 (Simulation Policy)
                # ここでは「現状維持」または「ランダム」を想定
                # より高度な実装では、Brainに想像上の観測を入力して行動を決定させることも可能

                # 例: ランダムな行動 (探索的な夢)
                action = torch.randn(
                    batch_size, self.action_dim, device=self.device)

                # 2. 次の状態を予測 (Predict Next)
                # WorldModelの predict_next メソッドを使用
                next_obs_pred = self.world_model.predict_next(
                    current_obs, action)

                # 3. 軌道の保存
                trajectory.append(next_obs_pred)

                # 4. 次のステップへの更新 (閉ループ)
                # predict_nextの結果は(B, D)になっている可能性があるため、Encoderが期待する形状(B, 1, D)等へ調整が必要
                # UniversalEncoderは次元数を見て判断するため、適切なリシェイプを行う

                # next_obs_predの各要素について、Encoderが受け入れ可能な形式に変換して次のcurrent_obsとする
                # ここでは簡易的に、WorldModelの出力がそのまま次の入力に使える特徴量空間にあると仮定、
                # もしくはUniversalEncoderを通す前の生データ形式に戻す必要がある（Decoderが必要）。

                # SpikingWorldModelの実装では decoders を持っており、predict_next は再構成された(B, D)を返す。
                # これを次の入力とするには、時間次元を追加する。
                current_obs = {k: v.unsqueeze(1)
                               for k, v in next_obs_pred.items()}

        return trajectory
