# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (DDP対応 & 最適化版)
# Description:
# - すべてのSNNモデルが継承する基底クラス。
# - 修正: get_total_spikes メソッドを分散学習 (DDP) に対応させ、
#   全デバイスのスパイク数を正しく集計するように修正。

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Optional

class SNNLayerNorm(nn.Module):
    """
    SNN用のLayerNorm。
    """
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """
    すべてのSNNモデルが継承する基底クラス。
    """
    def _init_weights(self) -> None:
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def get_total_spikes(self) -> float:
        """
        モデル全体の総スパイク数を計算する。
        分散学習時は全プロセスのスパイク数を合計する。
        """
        total_spikes_tensor = torch.tensor(0.0)
        
        # デバイスの特定（最初のパラメータがあればそれを使う）
        try:
            device = next(self.parameters()).device
            total_spikes_tensor = total_spikes_tensor.to(device)
        except StopIteration:
            pass # パラメータがない場合はCPUのまま

        # ローカルのスパイク数を集計
        for module in self.modules():
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                total_spikes_tensor += module.total_spikes
        
        # 分散学習環境での集計 (All-Reduce)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_spikes_tensor, op=dist.ReduceOp.SUM)
            # 平均発火率などの計算のために合計値が必要だが、
            # ここでは単純合計を返す。平均化が必要なら呼び出し元で行う。

        return total_spikes_tensor.item()
    
    def reset_spike_stats(self) -> None:
        """
        スパイク関連の統計情報をリセットする。
        """
        for module in self.modules():
            # spikingjelly準拠の reset()
            if hasattr(module, 'reset') and callable(module.reset):
                module.reset()
            
            # 明示的なカウンタリセット
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                with torch.no_grad():
                    module.total_spikes.zero_()
            
            if hasattr(module, 'spikes') and isinstance(module.spikes, torch.Tensor):
                with torch.no_grad():
                    module.spikes.zero_()