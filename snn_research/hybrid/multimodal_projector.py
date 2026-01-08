# ファイルパス: snn_research/hybrid/multimodal_projector.py
# 日本語タイトル: Multimodal Projector (Upgraded to MLP)
# 目的: 視覚特徴量を言語空間へより正確に射影するため、単層Linearから2層MLPへ強化。

import torch
import torch.nn as nn
from snn_research.core.base import BaseModel

class MultimodalProjector(BaseModel):
    """
    視覚モデルの出力を言語モデルの入力コンテキストとして射影するモジュール。
    [Update] 表現力向上のため MLP (Linear -> GELU -> Linear) を採用。
    """
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

        # ★強化ポイント: 中間層を設けて非線形変換を行う (MLP)
        hidden_dim = lang_dim * 4  # 一般的に隠れ層は出力の4倍程度
        
        if use_bitnet:
            from snn_research.training.quantization import BitLinear
            self.projector = nn.Sequential(
                BitLinear(visual_dim, hidden_dim, bias=False, weight_bits=1.58),
                nn.GELU(),
                BitLinear(hidden_dim, lang_dim, bias=False, weight_bits=1.58)
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim, bias=False),
                nn.GELU(), # 非線形活性化関数
                nn.Linear(hidden_dim, lang_dim, bias=False)
            )
            
        # 視覚トークンとしての位置エンコーディング
        self.pos_embed = nn.Parameter(torch.randn(1, 1, lang_dim) * 0.02)

        self._init_weights()

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        B = visual_features.shape[0]
        
        # 入力形状の正規化 -> (B, T_seq, C_vis)
        if visual_features.dim() == 2: # (B, C)
            x = visual_features.unsqueeze(1)
        elif visual_features.dim() == 3: # (B, T, C)
            x = visual_features
        elif visual_features.dim() == 4: # (B, C, H, W)
             B, C, H, W = visual_features.shape
             x = visual_features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            raise ValueError(f"Unsupported visual_features shape: {visual_features.shape}")

        # 射影実行 (MLP)
        x_proj = self.projector(x)
        
        # 位置エンコーディング加算
        x_proj = x_proj + self.pos_embed

        return x_proj