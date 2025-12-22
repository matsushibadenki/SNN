# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (Anyキャストによる動的呼び出し修正版)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class SNNCore(nn.Module):
    def __init__(self, config: Dict[str, Any], vocab_size: int = 1000):
        super().__init__()
        # 内部モデルは多様なアーキテクチャをとるため Any として定義し、mypyのメソッドチェックを回避
        from snn_research.core.architecture_registry import ArchitectureRegistry
        self.model: Any = ArchitectureRegistry.build(
            config.get('architecture_type', 'unknown'), config, vocab_size
        )

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """mypyエラー修正: Any型へのキャストにより、generate() の呼び出しを許可。"""
        return self.model.generate(input_ids, **kwargs)

    def reset_state(self) -> None:
        """mypyエラー修正: Any型へのキャストにより、reset_state() の呼び出しを許可。"""
        self.model.reset_state()

    def get_total_spikes(self) -> float:
        """mypyエラー修正: get_total_spikes() の呼び出しを許可。"""
        return float(self.model.get_total_spikes())

    def get_firing_rates(self) -> Dict[str, float]:
        """ActiveInferenceAgent からの呼び出しに対応。"""
        if hasattr(self.model, 'get_firing_rates'):
            return self.model.get_firing_rates()
        return {}
