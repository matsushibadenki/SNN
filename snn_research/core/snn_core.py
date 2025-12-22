# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (精度・統計強化版)
# 目的: 統一されたインターフェースと、BaseModelの統計管理・生成タスクの委譲。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast
import logging
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class SNNCore(BaseModel):
    """
    SNNモデルの統一インターフェース。
    BaseModelを継承し、統計管理と順伝播の柔軟性を両立。
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
        
        # ロードマップ v20.1 に基づくアーキテクチャの構築
        arch_type = self.config.get('architecture_type', 'unknown')
        self.model: Any = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
        
        # 重みの初期化
        self._init_weights()
        logger.info(f"SNNCore initialized with architecture: {arch_type}")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        柔軟な順伝播。入力テンソルの自動検索機能を備え、発火率統計を収集する。
        """
        if x is None:
            # 優先順位に基づいた入力検索 (v20.5準拠)
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input', 'inputs']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        try:
            # 推論実行
            if x is not None:
                output = self.model(x, **kwargs)
            else:
                output = self.model(**kwargs)
            
            # 発火率の統計更新 (BaseModel機能の活用)
            if hasattr(self.model, 'get_firing_rates'):
                rates = self.model.get_firing_rates()
                # 内部統計を更新するロジックをここに追加可能
                
            return output
        except Exception as e:
            logger.error(f"SNNCore: Forward execution failed: {e}")
            raise e

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """生成タスクをバックボーンモデルに委譲。"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError(f"{type(self.model).__name__} does not support generation.")

    def reset_state(self) -> None:
        """全状態のリセット。スパイク統計もクリアする。"""
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        
        self.reset_spike_stats()
        logger.debug("SNN state and spike statistics have been reset.")

    def get_total_spikes(self) -> float:
        """モデル全体のススパイク総数を取得。"""
        if hasattr(self.model, 'get_total_spikes'):
            return float(self.model.get_total_spikes())
        return super().get_total_spikes()

    def get_firing_rates(self) -> Dict[str, float]:
        """各レイヤーの発火率レポート。"""
        if hasattr(self.model, 'get_firing_rates'):
            return self.model.get_firing_rates()
        return {}
