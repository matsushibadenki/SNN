# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (ランタイムエラー修正版)
# 目的: SNNCore の初期化引数不一致および forward 未実装エラーを解消する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast
import logging

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNモデルの統一インターフェース。
    mypyエラーおよびランタイムの NotImplementedError を解消。
    """
    def __init__(
        self, 
        config: Dict[str, Any], 
        vocab_size: int = 1000, 
        backend: str = "spikingjelly" # backend 引数を追加して初期化エラーを修正
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.backend = backend
        
        from snn_research.core.architecture_registry import ArchitectureRegistry
        
        # 内部モデルをビルド
        arch_type = self.config.get('architecture_type', 'unknown')
        self.model: Any = ArchitectureRegistry.build(arch_type, self.config, vocab_size)

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        柔軟な順伝播。多種多様な引数名に対応し、NotImplementedError を回避。
        """
        if x is None:
            # 優先度の高い順に入力テンソルを検索
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input', 'inputs']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        # モデルの実行
        try:
            if x is not None:
                return self.model(x, **kwargs)
            else:
                return self.model(**kwargs)
        except Exception as e:
            logger.error(f"SNNCore: Forward execution failed: {e}")
            raise e

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError(f"{type(self.model).__name__} does not support generation.")

    def reset_state(self) -> None:
        """モデル内の全状態（SSMキャッシュ、膜電位等）をリセット。"""
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)

    def get_total_spikes(self) -> float:
        if hasattr(self.model, 'get_total_spikes'):
            return float(self.model.get_total_spikes())
        return 0.0

    def get_firing_rates(self) -> Dict[str, float]:
        if hasattr(self.model, 'get_firing_rates'):
            return self.model.get_firing_rates()
        return {}
