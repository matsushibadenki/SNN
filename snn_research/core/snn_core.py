# /snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (精度・統計強化版)
# 目的: BaseModelの統計管理機能を正しく呼び出し、mypyエラーを解消する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast
import logging
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class SNNCore(BaseModel):
    """
    SNNモデルの統一インターフェース。
    """
    def __init__(
        self, 
        config: Dict[str, Any], 
        vocab_size: int = 1000, 
        backend: str = "spikingjelly"
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.backend = backend
        
        from snn_research.core.architecture_registry import ArchitectureRegistry
        
        arch_type = self.config.get('architecture_type', 'unknown')
        self.model: nn.Module = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
        
        self._init_weights()
        # 統計保持用の辞書を初期化
        self.spike_stats: Dict[str, float] = {}

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if x is None:
            # 入力候補を探索
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        try:
            if x is not None:
                output = self.model(x, **kwargs)
            else:
                output = self.model(**kwargs)
            
            # 推論後の統計更新
            self._update_firing_stats()
            return output
        except Exception as e:
            logger.error(f"SNNCore: Forward failed: {e}")
            raise e

    def _update_firing_stats(self) -> None:
        """発火率統計を安全に更新する。"""
        # モデル自体が get_firing_rates を持っている場合、または子レイヤーを走査
        if hasattr(self.model, 'get_firing_rates'):
            rates = self.model.get_firing_rates()
            if isinstance(rates, dict):
                for layer_name, rate in rates.items():
                    val = float(rate.item()) if isinstance(rate, torch.Tensor) else float(rate)
                    self.spike_stats[layer_name] = val
        else:
            # 各レイヤーを走査して統計を集計（LIFLayerなどの個別の層から取得）
            for name, module in self.model.named_modules():
                if hasattr(module, 'get_firing_rate'):
                    self.spike_stats[name] = module.get_firing_rate()

    def get_firing_rates(self) -> Dict[str, float]:
        """外部から統計を取得するためのメソッド。"""
        return self.spike_stats

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError(f"{type(self.model).__name__} does not support generation.")

    def reset_state(self) -> None:
        """ネットワーク状態（膜電位等）と統計をリセット。"""
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        self.spike_stats.clear()
        logger.debug("SNN state and statistics reset.")

    def _init_weights(self) -> None:
        """重みの初期化。"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
