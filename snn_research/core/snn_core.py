# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (Refactored)
# ファイルの目的・内容:
#   各種SNNモデルの統一インターフェースを提供するラッパー。
#   ArchitectureRegistryとの連携を整理し、循環参照を回避する。
#   共通のユーティリティ（リセット、スパイク集計）を提供する。

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
        
        # モデル構築
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
        柔軟な順伝播。x が None の場合は kwargs から入力を探索する。
        """
        # 入力テンソルの特定
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        # モデルへの委譲
        if x is not None:
            return self.model(x, **kwargs)
        else:
            # 入力が特定できない場合は kwargs 全体を渡す（モデル側で処理）
            return self.model(**kwargs)
    
    def reset_state(self) -> None:
        """状態リセットの統一インターフェース"""
        # 1. SpikingJellyの標準リセット
        functional.reset_net(self.model)

        # 2. カスタムリセット (reset_state メソッドを持つ場合)
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
