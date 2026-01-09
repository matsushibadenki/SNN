# ファイルパス: snn_research/hybrid/multimodal_projector.py
# 日本語タイトル: Unified Sensory Projector (Multi-Modal Adapter)
# 修正内容: 任意の感覚入力を統合するUnifiedSensoryProjectorの実装と、旧MultimodalProjectorへの互換性クラスの追加。

import torch
import torch.nn as nn
from typing import Dict
from snn_research.core.base import BaseModel


class UnifiedSensoryProjector(BaseModel):
    """
    あらゆる感覚モダリティの特徴量を、言語モデル(Brain)が理解可能な埋め込み空間へ射影する統合モジュール。
    Vision, Audio, Tactile などを動的に辞書形式で受け取り、統合されたコンテキスト列を生成する。
    """

    def __init__(
        self,
        language_dim: int,
        # Example: {'vision': 128, 'audio': 64}
        modality_configs: Dict[str, int],
        use_bitnet: bool = False
    ):
        super().__init__()
        self.language_dim = language_dim
        self.modality_configs = modality_configs

        # モダリティごとのアダプターを動的に生成
        self.adapters = nn.ModuleDict()
        self.pos_embeds = nn.ParameterDict()

        for modality, input_dim in modality_configs.items():
            # 各感覚用の射影ネットワーク (MLP)
            self.adapters[modality] = self._build_mlp(
                input_dim, language_dim, use_bitnet)

            # 各感覚固有の「感覚タグ」としての学習可能な埋め込み
            # これにより脳は「これは映像由来」「これは音由来」と識別可能になる
            self.pos_embeds[modality] = nn.Parameter(
                torch.randn(1, 1, language_dim) * 0.02)

        # 統合後のゲート機構 (感覚間の重み付け用)
        self.sensory_gate = nn.Sequential(
            nn.Linear(language_dim, language_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _build_mlp(self, input_dim: int, output_dim: int, use_bitnet: bool) -> nn.Module:
        """射影用MLPの構築"""
        hidden_dim = output_dim * 4

        if use_bitnet:
            from snn_research.training.quantization import BitLinear
            return nn.Sequential(
                BitLinear(input_dim, hidden_dim, bias=False, weight_bits=1.58),
                nn.GELU(),
                BitLinear(hidden_dim, output_dim, bias=False, weight_bits=1.58)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=False)
            )

    def forward(self, sensory_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            sensory_inputs: モダリティ名をキーとする特徴量テンソルの辞書
                           Example: {'vision': (B, T, C), 'audio': (B, T, C)}
        Returns:
            projected_context: (B, Total_Time, Language_Dim)
        """
        # バッチサイズの取得（入力が空の場合はデフォルト1とするが、通常は呼び出し側で制御）
        if not sensory_inputs:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.language_dim, device=device)

        batch_size = next(iter(sensory_inputs.values())).shape[0]
        projected_features = []

        # 定義されたモダリティ順に処理 (順序を固定することで学習を安定化)
        for modality in self.modality_configs.keys():
            if modality not in sensory_inputs:
                continue

            features = sensory_inputs[modality]

            # 入力形状の正規化 -> (B, T, C_in)
            if features.dim() == 2:  # (B, C) -> (B, 1, C)
                x = features.unsqueeze(1)
            elif features.dim() == 3:  # (B, T, C)
                x = features
            elif features.dim() == 4:  # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = features.shape
                x = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            elif features.dim() == 5:  # (B, T, C, H, W) -> Video
                B, T, C, H, W = features.shape
                x = features.permute(0, 1, 3, 4, 2).reshape(B, T*H*W, C)
            else:
                raise ValueError(
                    f"Unsupported shape for {modality}: {features.shape}")

            # 射影
            if modality in self.adapters:
                x_proj = self.adapters[modality](x)
                if modality in self.pos_embeds:
                    x_proj = x_proj + self.pos_embeds[modality]
                projected_features.append(x_proj)

        if not projected_features:
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 1, self.language_dim, device=device)

        # 時間方向に結合 [Vision Tokens, Audio Tokens, ...]
        combined_context = torch.cat(projected_features, dim=1)

        # ゲート通過
        gate = self.sensory_gate(combined_context)
        return combined_context * gate

# --- Backward Compatibility ---


class MultimodalProjector(UnifiedSensoryProjector):
    """
    旧コード(Brain v4初期版など)との互換性を保つためのラッパー。
    単一入力(主にVision)を想定したインターフェースを提供する。
    """

    def __init__(
        self,
        visual_dim: int,
        lang_dim: int,
        visual_time_steps: int = 16,
        lang_time_steps: int = 16,
        use_bitnet: bool = False
    ):
        # 内部的には 'legacy_input' というモダリティとして扱う
        super().__init__(
            language_dim=lang_dim,
            modality_configs={'legacy_input': visual_dim},
            use_bitnet=use_bitnet
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # 辞書形式にラップして親クラスに渡す
        return super().forward({'legacy_input': x})
