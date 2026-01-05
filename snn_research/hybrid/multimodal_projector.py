# ファイルパス: snn_research/hybrid/multimodal_projector.py
# (修正: 循環インポート回避のための遅延インポート)

import torch
import torch.nn as nn

from snn_research.core.base import BaseModel
# 削除: from snn_research.training.quantization import BitLinear  <-- ここを削除

class MultimodalProjector(BaseModel):
    """
    視覚モデルの出力を言語モデルの入力コンテキストとして射影するモジュール。
    """
    projector: nn.Linear

    def __init__(
        self,
        visual_dim: int,
        lang_dim: int,
        visual_time_steps: int,
        lang_time_steps: int,
        use_bitnet: bool = False
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.lang_dim = lang_dim
        self.visual_time_steps = visual_time_steps
        self.lang_time_steps = lang_time_steps

        # 1. 次元射影層 (Visual Dim -> Language Dim)
        if use_bitnet:
            # --- 追加: ここでインポート ---
            from snn_research.training.quantization import BitLinear
            # BitLinear は nn.Linear を継承しているため、nn.Linear 型として扱える
            self.projector = BitLinear(visual_dim, lang_dim, bias=False, weight_bits=1.58)
        else:
            self.projector = nn.Linear(visual_dim, lang_dim, bias=False)
            
        # 2. オプション: 視覚トークンとしての位置エンコーディング
        # (シーケンスとして扱う場合)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, lang_dim) * 0.02)

        self._init_weights()

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (Batch, T_vis, C_vis) または (Batch, C_vis)
                             SpikingCNNの出力（平均スパイク率または最終膜電位）

        Returns:
            projected_context: (Batch, Context_Len, Lang_Dim)
                               言語モデルに注入するためのコンテキスト埋め込み
        """
        B = visual_features.shape[0]
        
        # 入力形状の正規化 -> (B, T_seq, C_vis)
        if visual_features.dim() == 2: # (B, C)
            x = visual_features.unsqueeze(1) # (B, 1, C)
        elif visual_features.dim() == 3: # (B, T, C)
            x = visual_features
        elif visual_features.dim() == 4: # (B, C, H, W) -> Flatten -> (B, H*W, C) or Pool
             # 空間情報をシーケンスとして扱う (ViTライク)
             B, C, H, W = visual_features.shape
             x = visual_features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            raise ValueError(f"Unsupported visual_features shape: {visual_features.shape}")

        # 1. 次元射影
        # x: (B, Seq, C_vis) -> (B, Seq, Lang_Dim)
        x_proj = self.projector(x)
        
        # 2. 時間軸/シーケンス長の調整 (必要な場合)
        # ここでは視覚特徴のシーケンス長をそのままコンテキスト長として扱う
        # 時間的なリサンプリングが必要な場合はここで interpolate する
        
        # 位置エンコーディングの加算 (ブロードキャスト)
        x_proj = x_proj + self.pos_embed

        return x_proj