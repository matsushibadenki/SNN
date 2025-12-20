# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー v16.3 (Generator Support)
# 目的・内容:
#   各種SNNモデルの統一インターフェース。
#   ReasoningEngineからの利用を想定し、generateメソッドの委譲処理を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
from spikingjelly.activation_based import functional # type: ignore

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNモデルの統一ラッパークラス。
    すべてのモデルはここを経由して初期化・実行されることで、
    入出力のインターフェースや状態管理が統一される。
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
        
        # 遅延インポートで循環参照を回避
        from snn_research.core.architecture_registry import ArchitectureRegistry
        
        try:
            self.model: nn.Module = ArchitectureRegistry.build(
                self.config.get('architecture_type', 'unknown'),
                self.config,
                self.vocab_size
            )
        except ValueError as e:
            logger.error(f"Model building failed: {e}")
            raise e

        self._log_model_stats()

    def _log_model_stats(self):
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"✅ SNNCore initialized. Params: {param_count:,} (Trainable: {trainable:,})")
        except Exception:
            pass

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        柔軟かつ厳格な順伝播。
        x が None の場合は、優先順位リストに従って kwargs から入力を探索する。
        """
        if x is None:
            # 優先順位リスト: LLM系(input_ids) > 画像系(input_images) > 汎用(x, input)
            prioritized_keys = ['input_ids', 'input_images', 'x', 'input_sequence', 'input']
            for key in prioritized_keys:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        if x is None and not kwargs:
             logger.debug("SNNCore forward called with no explicit input tensor. Assuming internal state generation.")

        if x is not None:
            return self.model(x, **kwargs)
        else:
            return self.model(**kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        生成タスク（テキスト生成など）を内部モデルに委譲する。
        ReasoningEngineやSystem 1の推論で使用される。
        """
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            return self.model.generate(input_ids, **kwargs) # type: ignore
        else:
            raise NotImplementedError(
                f"The underlying model '{type(self.model).__name__}' does not support 'generate' method."
            )
    
    def reset_state(self) -> None:
        """状態リセットの統一インターフェース"""
        # 1. SpikingJellyの標準リセット
        functional.reset_net(self.model)

        # 2. カスタムリセット
        if hasattr(self.model, 'reset_state') and callable(getattr(self.model, 'reset_state')):
             getattr(self.model, 'reset_state')()
        
        # 3. 統計リセット
        if hasattr(self.model, 'reset_spike_stats'):
             getattr(self.model, 'reset_spike_stats')()

    def get_total_spikes(self) -> float:
        """総スパイク数の取得"""
        if hasattr(self.model, 'get_total_spikes'):
            return self.model.get_total_spikes() # type: ignore
        return 0.0