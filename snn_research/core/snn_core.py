# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (型呼び出し修正版)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class SNNCore(nn.Module):
    def __init__(self, config: Dict[str, Any], vocab_size: int = 1000):
        super().__init__()
        # ArchitectureRegistry.build() は nn.Module を返すが、
        # generate() 等のメソッドを動的に持つため Any で管理
        from snn_research.core.architecture_registry import ArchitectureRegistry
        self.model: Any = ArchitectureRegistry.build(
            config.get('architecture_type', 'unknown'), config, vocab_size
        )

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """mypyエラー修正: model を Any として扱うことで Tensor 非呼び出しエラーを回避。"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError("Internal model lacks 'generate' method.")

    def get_firing_rates(self) -> Dict[str, float]:
        """ActiveInferenceAgent 等が期待するメソッド。"""
        if hasattr(self.model, 'get_firing_rates'):
            return self.model.get_firing_rates()
        return {}
