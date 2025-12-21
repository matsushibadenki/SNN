# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー v20.5 (Enhanced Async Compatibility)
# 目的・内容:
#   各種SNNモデル（SFormer, BitSpikeMamba等）の統一インターフェース。
#   - 非同期カーネル（ArtificialBrain）からの動的な入力を処理するための柔軟なフォワードロジック。
#   - Bit-Spike量子化モデルの統計収集（発火率監視）をサポート。
#   - 生成タスクにおける例外処理の強化。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from spikingjelly.activation_based import functional # type: ignore

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNモデルの統一ラッパークラス。
    すべてのモデル（System 1等）はここを経由して初期化・実行されることで、
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
            # ロードマップ v20.0 に基づく BitSpike モデル等のビルド
            self.model: nn.Module = ArchitectureRegistry.build(
                self.config.get('architecture_type', 'unknown'),
                self.config,
                self.vocab_size
            )
        except ValueError as e:
            logger.error(f"❌ SNNCore: Model building failed: {e}")
            raise e

        self._log_model_stats()

    def _log_model_stats(self):
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            arch_type = self.config.get('architecture_type', 'unknown')
            logger.info(f"✅ SNNCore initialized [{arch_type}]. Params: {param_count:,} (Trainable: {trainable:,})")
        except Exception:
            pass

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        柔軟かつ厳格な順伝播。
        非同期バス経由の多様な入力（画像、テキスト、時系列）を自動識別する。
        """
        if x is None:
            # ロードマップのマルチモーダル統合に対応する優先順位
            # LLM(input_ids) > 画像(input_images) > 生体信号(input_sequence) > 汎用
            prioritized_keys = ['input_ids', 'input_images', 'input_sequence', 'x', 'input']
            for key in prioritized_keys:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        if x is None and not kwargs:
             logger.debug("SNNCore: Forward called with no explicit input. Triggering autonomous state update.")

        # モデル実行
        if x is not None:
            return self.model(x, **kwargs)
        else:
            return self.model(**kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        生成タスク（System 2からの思考委譲）を内部モデルに委譲。
        BitSpikeMamba等の自己回帰モデルで使用される。
        """
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            try:
                return self.model.generate(input_ids, **kwargs) # type: ignore
            except Exception as e:
                logger.error(f"❌ SNNCore: Generation error: {e}")
                raise e
        else:
            model_name = type(self.model).__name__
            raise NotImplementedError(
                f"The underlying model '{model_name}' does not support 'generate' method. "
                "Ensure System 1 backbone is an autoregressive SNN."
            )
    
    def reset_state(self) -> None:
        """状態リセットの統一インターフェース。AstrocyteNetworkの睡眠サイクル等から呼ばれる。"""
        # 1. SpikingJellyの標準リセット（膜電位等）
        functional.reset_net(self.model)

        # 2. カスタムリセット（MambaのSSM状態やキャッシュ等）
        if hasattr(self.model, 'reset_state') and callable(getattr(self.model, 'reset_state')):
             getattr(self.model, 'reset_state')()
        
        # 3. 統計リセット（発火率、エネルギー消費統計）
        if hasattr(self.model, 'reset_spike_stats') and callable(getattr(self.model, 'reset_spike_stats')):
             getattr(self.model, 'reset_spike_stats')()

    def get_total_spikes(self) -> float:
        """
        総スパイク数の取得。
        AstrocyteNetworkがエネルギー消費を監視するために使用。
        """
        if hasattr(self.model, 'get_total_spikes') and callable(getattr(self.model, 'get_total_spikes')):
            return float(self.model.get_total_spikes()) # type: ignore
        return 0.0

    def get_firing_rates(self) -> Dict[str, float]:
        """各層の平均発火率を取得（デバッグおよびメタ認知用）"""
        if hasattr(self.model, 'get_firing_rates') and callable(getattr(self.model, 'get_firing_rates')):
            return self.model.get_firing_rates() # type: ignore
        return {}
