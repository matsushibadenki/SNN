# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (リファクタリング版)
# Description: DDP対応のスパイク集計、標準的なPyTorch状態管理、および高度な重み初期化ロジックの統合。

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Dict

class SNNLayerNorm(nn.Module):
    """SNNのスパイク活動または膜電位に適応するLayerNorm。"""
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """
    すべてのSNNモデルの基盤となる抽象クラス。
    分散学習(DDP)環境でのスパイク統計の同期と、再帰的な状態管理を提供します。
    """
    def __init__(self):
        super().__init__()

    def _init_weights(self) -> None:
        """標準的な初期化戦略。各サブクラスで必要に応じてオーバーライド可能。"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                # SNN特有のダイナミクスを考慮し、Kaiming Normalを適用
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def get_total_spikes(self) -> float:
        """
        全レイヤーおよび全計算ノード(DDP)の総スパイク数を集計。
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        total = torch.tensor(0.0, device=device)
        
        for m in self.modules():
            ts = getattr(m, 'total_spikes', None)
            if isinstance(ts, torch.Tensor):
                # デバイスを合わせ、テンソル全体の合計を加算
                total += ts.to(device).sum()
        
        # 分散学習環境での同期
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        return total.item()
    
    def reset_spike_stats(self) -> None:
        """
        ネットワーク全体の統計情報と膜電位を再帰的にリセット。
        """
        for m in self.modules():
            if hasattr(m, 'reset') and callable(m.reset):
                m.reset()
            
            for attr in ['total_spikes', 'spikes']:
                val = getattr(m, attr, None)
                if isinstance(val, torch.Tensor):
                    val.detach().zero_()
                    
    def get_model_metrics(self) -> Dict[str, float]:
        """モデルの健康状態や活動レベルを辞書形式で返す。"""
        return {
            "total_spikes": self.get_total_spikes(),
            "param_count": float(sum(p.numel() for p in self.parameters()))
        }