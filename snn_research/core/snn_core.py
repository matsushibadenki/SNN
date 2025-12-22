# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー v20.6 (Full State Integration)
# 目的: あらゆるSNNアーキテクチャに対して、統一された生成・リセット・統計インターフェースを提供。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from spikingjelly.activation_based import functional

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNバックボーンの抽象化レイヤー。
    System 1（直感）と System 2（論理）の橋渡しを行う。
    """
    def __init__(self, config: Dict[str, Any], vocab_size: int = 1000):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        from snn_research.core.architecture_registry import ArchitectureRegistry
        
        arch_type = self.config.get('architecture_type', 'bit_spike_mamba')
        try:
            self.model = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
            logger.info(f"🚀 SNNCore: Architecture '{arch_type}' successfully deployed.")
        except Exception as e:
            logger.error(f"❌ SNNCore: Failed to build architecture '{arch_type}': {e}")
            raise

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        # 入力の自動解決
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        # モデルへの委譲
        return self.model(x, **kwargs) if x is not None else self.model(**kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """自己回帰生成の委譲。"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError(f"Model {type(self.model).__name__} does not support sequence generation.")

    def reset_state(self) -> None:
        """
        全状態の完全リセット。Astrocyte（星状細胞）による睡眠・リフレッシュ時に重要。
        """
        # 1. 膜電位のリセット
        functional.reset_net(self.model)
        
        # 2. モデル固有の状態（SSMの隠れ状態等）のリセット
        reset_fn = getattr(self.model, 'reset_state', None)
        if callable(reset_fn):
            reset_fn()
            
        # 3. スパイク統計のリセット
        stats_reset_fn = getattr(self.model, 'reset_spike_stats', None)
        if callable(stats_reset_fn):
            stats_reset_fn()
        
        logger.debug(f"🧠 {type(self.model).__name__}: All internal states synchronized and reset.")

    def get_total_spikes(self) -> float:
        """エネルギー効率計算のためのスパイク集計。"""
        method = getattr(self.model, 'get_total_spikes', None)
        return float(method()) if callable(method) else 0.0
