# ファイルパス: snn_research/core/snn_core.py
# ファイル名: SNNコア・ラッパー (Input Handling Fix)
# 機能説明: 各種アーキテクチャ（CNN, Transformer, PCなど）を統一的にラップするクラス。
#          ArchitectureRegistryを使用して設定からモデルを構築し、
#          共通のインターフェース（forward, reset_state）を提供する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
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
        
        # パラメータ数の集計とログ出力（安全に実行）
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except Exception:
            param_count = 0
            trainable_count = 0
        
        arch_type = self.config.get('architecture_type', 'unknown')
        if param_count == 0:
            logger.warning(f"⚠️ Built model '{arch_type}' appears to have 0 parameters. This might be intentional (e.g. functional model) or an error.")
        else:
            logger.info(f"✅ SNNCore built model '{arch_type}' with {param_count:,} parameters ({trainable_count:,} trainable).")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        順伝播。入力テンソル x が None の場合、kwargs から適切なキーを探す。
        """
        # x が指定されていない場合、kwargs から入力テンソルを探す
        if x is None:
            # 優先順位の高いキーから順に探す
            # 注意: pop() すると下流のモデルがその引数を必要とする場合にエラーになる可能性があるため、
            # ここでは取得するだけにして、kwargs はそのままモデルに渡す設計にする。
            # ただし、x として渡す場合は kwargs から除外しないと重複エラーになる場合がある。
            # モデルの実装 (forward) 次第だが、一般的には位置引数 x があればそれを使い、
            # なければ kwargs を使う形が多い。
            
            found_key = None
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input']:
                if key in kwargs:
                    x = kwargs[key]
                    found_key = key
                    break
            
            # キーが見つかった場合、kwargs から削除して x として渡すことで重複を防ぐ
            if found_key:
                kwargs.pop(found_key)

        # モデルに渡す
        if x is None:
            # x が見つからなかった場合は kwargs のみで呼び出す
            return self.model(**kwargs)
        else:
            # x と残りの kwargs で呼び出す
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
