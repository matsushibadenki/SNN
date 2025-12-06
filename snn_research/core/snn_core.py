# ファイルパス: snn_research/core/snn_core.py
# Title: SNNCore (Registry Pattern Integrated & SpikingJelly Compatible)
# Description:
# - モデル構築ロジックを ArchitectureRegistry に移譲。
# - 設定管理と共通インターフェース（forward, reset_state）を提供。
# - 修正: SpikingJellyの再帰的リセットに対応するため、reset() メソッドを追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
# mypyエラー抑制
from spikingjelly.activation_based import functional # type: ignore

# レジストリをインポート
from .architecture_registry import ArchitectureRegistry

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
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
        
        self.model: nn.Module = self._build_model()
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        arch_type = self.config.get('architecture_type', 'unknown')
        if param_count == 0:
            logger.error(f"❌ Built model '{arch_type}' has 0 parameters! Check model initialization.")
        else:
            logger.info(f"✅ SNNCore built model '{arch_type}' with {param_count:,} parameters ({trainable_count:,} trainable).")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if x is None:
            # 辞書キーから入力を探す
            for key in ['input_ids', 'input_images', 'input_sequence', 'x']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        if x is None:
            return self.model(**kwargs)
        
        return self.model(x, **kwargs)
    
    def reset_state(self) -> None:
        """モデルの内部状態をリセットする。"""
        functional.reset_net(self.model)

        if hasattr(self.model, 'reset_state') and callable(getattr(self.model, 'reset_state')):
             getattr(self.model, 'reset_state')()
        
        if hasattr(self.model, 'reset_spike_stats'):
             getattr(self.model, 'reset_spike_stats')()

    def reset(self) -> None:
        """
        SpikingJelly互換のリセットメソッド。
        functional.reset_net() は再帰的に .reset() を呼び出すため、これが必要。
        """
        self.reset_state()

    def get_total_spikes(self) -> float:
        """内部モデルのスパイク総数を取得する。"""
        if hasattr(self.model, 'get_total_spikes'):
            return self.model.get_total_spikes() # type: ignore
        return 0.0

    def _build_model(self) -> nn.Module:
        """
        ArchitectureRegistry を使用してモデルを構築する。
        """
        arch_type = self.config.get('architecture_type')
        
        if not arch_type:
            logger.error(f"SNNCore Config keys: {list(self.config.keys())}")
            if 'model' in self.config:
                logger.error(f"Did you mean to pass config['model']? Found 'model' key in config.")
            raise ValueError("SNNCore: 'architecture_type' is missing in the configuration. Cannot build model.")

        if self.backend != "spikingjelly":
             raise ValueError(f"Unsupported backend: {self.backend}. Only 'spikingjelly' is supported.")

        # レジストリに委譲
        try:
            return ArchitectureRegistry.build(arch_type, self.config, self.vocab_size)
        except ValueError as e:
            # レジストリに未登録の場合のエラーハンドリング
            logger.error(f"Failed to build model: {e}")
            raise e
