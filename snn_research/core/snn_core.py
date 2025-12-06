# ファイルパス: snn_research/core/snn_core.py
# ファイル名: SNNコア・ラッパー
# 機能説明: 各種アーキテクチャ（CNN, Transformer, PCなど）を統一的にラップするクラス。
#          ArchitectureRegistryを使用して設定からモデルを構築し、
#          共通のインターフェース（forward, reset_state）を提供する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
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
        
        # レジストリ経由でモデルを構築
        self.model: nn.Module = self._build_model()
        
        # パラメータ数の集計とログ出力
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        arch_type = self.config.get('architecture_type', 'unknown')
        if param_count == 0:
            logger.error(f"❌ Built model '{arch_type}' has 0 parameters! Check model initialization.")
        else:
            logger.info(f"✅ SNNCore built model '{arch_type}' with {param_count:,} parameters ({trainable_count:,} trainable).")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        順伝播。入力テンソル x が None の場合、kwargs から適切なキーを探す。
        """
        if x is None:
            # 辞書キーから入力を探す (input_ids, input_images など)
            for key in ['input_ids', 'input_images', 'input_sequence', 'x']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        # モデルに渡す。xがまだNoneならkwargsのみで呼び出す。
        if x is None:
            return self.model(**kwargs)
        
        return self.model(x, **kwargs)
    
    def reset_state(self) -> None:
        """モデルの内部状態（膜電位、スパイク履歴など）をリセットする。"""
        # SpikingJellyの標準リセット
        functional.reset_net(self.model)

        # カスタムリセットメソッドがあれば呼ぶ
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

        # レジストリに構築を委譲
        try:
            return ArchitectureRegistry.build(arch_type, self.config, self.vocab_size)
        except ValueError as e:
            logger.error(f"Failed to build model: {e}")
            raise e
