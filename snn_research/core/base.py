# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (パフォーマンス最適化版)
# Description:
# - すべてのSNNモデルが継承する基底クラス。
# - 修正: get_total_spikes メソッドでの頻繁な GPU-CPU 同期 (.item()) を廃止。
#   Tensorとして合計し、最後に1回だけ同期することで、学習・推論速度を向上させる。

import torch
import torch.nn as nn
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
        最適化: Tensorとして加算し、最後に1回だけCPUへ転送(.item())する。
        """
        total_spikes_tensor = torch.tensor(0.0)
        
        # デバイスの特定（最初のパラメータがあればそれを使う）
        try:
            device = next(self.parameters()).device
            total_spikes_tensor = total_spikes_tensor.to(device)
        except StopIteration:
            pass # パラメータがない場合はCPUのまま

        for module in self.modules():
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                # 加算 (GPU上で行われる)
                total_spikes_tensor += module.total_spikes
        
        # 最後に1回だけ同期して値を返す
        return total_spikes_tensor.item()
    
    def reset_spike_stats(self) -> None:
        """
        スパイク関連の統計情報をリセットする。
        """
        for module in self.modules():
            # spikingjelly準拠の reset()
            if hasattr(module, 'reset') and callable(module.reset):
                module.reset()
            
            # 明示的なカウンタリセット (reset() でクリアされない実装への保険)
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                with torch.no_grad():
                    module.total_spikes.zero_()
            
            if hasattr(module, 'spikes') and isinstance(module.spikes, torch.Tensor):
                with torch.no_grad():
                    module.spikes.zero_()
