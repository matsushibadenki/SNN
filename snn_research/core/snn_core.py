# /snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (統計・動的推論強化版)
# 目的: 各種SNNアーキテクチャを統一インターフェースで包み、発火統計と生成タスクを管理する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class SNNCore(BaseModel):
    """
    SNNモデルの統一インターフェース。
    ニューロモーフィックOSにおける「アプリケーション」と「ハードウェア」の橋渡しを担う。
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
        
        # レジストリからアーキテクチャをビルド
        arch_type = self.config.get('architecture_type', 'sformer')
        self.model: Any = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
        
        # 重みの初期化とデバイス転送
        self._init_weights()
        logger.info(f"SNNCore: [{arch_type}] backend:[{backend}] initialized.")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        柔軟な順伝播。入力の自動検知とスパイク統計の収集を行う。
        """
        if x is None:
            # kwargs内から適切な入力を検索
            input_keys = ['input_ids', 'input_images', 'input_sequence', 'x', 'input']
            for key in input_keys:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        try:
            # 実際の推論実行
            if x is not None:
                output = self.model(x, **kwargs)
            else:
                output = self.model(**kwargs)
            
            # 発火率統計の自動収集
            self._update_firing_stats()
                
            return output
        except Exception as e:
            logger.error(f"SNNCore: Forward failed: {str(e)}")
            raise e

    def _update_firing_stats(self) -> None:
        """モデル内の各レイヤーから発火データを取得し、BaseModelの統計を更新する。"""
        if hasattr(self.model, 'get_firing_rates'):
            rates = self.model.get_firing_rates()
            if isinstance(rates, dict):
                for layer_name, rate in rates.items():
                    # 基底クラスの統計管理機能を利用（仮想的な実装想定）
                    if hasattr(self, 'spike_stats'):
                        self.spike_stats[layer_name] = float(rate)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """言語生成タスクなどをバックボーンモデルに委譲。"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids, **kwargs)
        raise NotImplementedError(f"Generation not supported by {type(self.model).__name__}")

    def reset_state(self) -> None:
        """膜電位およびスパイク統計の完全リセット。"""
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        
        # BaseModelの統計リセット
        if hasattr(self, 'reset_spike_stats'):
            self.reset_spike_stats()
        logger.debug("SNNCore: All states and statistics reset.")

    def get_total_spikes(self) -> float:
        """モデル全体の累積スパイク数を取得。"""
        if hasattr(self.model, 'get_total_spikes'):
            return float(self.model.get_total_spikes())
        return 0.0

    def _init_weights(self) -> None:
        """生物学的な妥当性を考慮した重みの初期化。"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
