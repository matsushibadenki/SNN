# /snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (mypy完全適合版)
# 目的: "Tensor not callable" エラーを回避するため、動的属性アクセスとCallableチェックを徹底する。

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
        # mypyの誤認を防ぐため、敢えてAnyのまま安全に取り扱う
        self.model: Any = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
        
        self._init_weights()
        self.spike_stats: Dict[str, float] = {}

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        try:
            # self.model(x) の呼び出し自体は nn.Module として許容される
            if x is not None:
                output = self.model(x, **kwargs)
            else:
                output = self.model(**kwargs)
            
            self._update_firing_stats()
            return output
        except Exception as e:
            logger.error(f"SNNCore: Forward failed: {e}")
            raise e

    def _update_firing_stats(self) -> None:
        """発火率統計を安全に更新する。"""
        # [修正] 直接ドットでアクセスせず、getattrとCallableチェックを使用する
        get_rates_method = getattr(self.model, 'get_firing_rates', None)
        
        if get_rates_method is not None and callable(get_rates_method):
            rates = get_rates_method()
            if isinstance(rates, dict):
                for layer_name, rate in rates.items():
                    val = float(rate.item()) if isinstance(rate, torch.Tensor) else float(rate)
                    self.spike_stats[layer_name] = val
        else:
            # 代替案として子モジュールの get_firing_rate を探索
            for name, module in self.model.named_modules():
                child_rate_method = getattr(module, 'get_firing_rate', None)
                if child_rate_method is not None and callable(child_rate_method):
                    self.spike_stats[name] = child_rate_method()

    def get_firing_rates(self) -> Dict[str, float]:
        """外部から統計を取得するためのメソッド。"""
        return self.spike_stats

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """[修正] generateメソッドを安全に呼び出す。"""
        gen_method = getattr(self.model, 'generate', None)
        if gen_method is not None and callable(gen_method):
            return cast(torch.Tensor, gen_method(input_ids, **kwargs))
            
        raise NotImplementedError(f"{type(self.model).__name__} does not support generation.")

    def reset_state(self) -> None:
        """[修正] ネットワーク状態と統計をリセットする。"""
        from spikingjelly.activation_based import functional
        # spikingjellyの関数は型チェックを通る
        functional.reset_net(self.model)
        
        # モデル独自のreset_stateを安全に呼び出す
        reset_method = getattr(self.model, 'reset_state', None)
        if reset_method is not None and callable(reset_method):
            reset_method()
            
        self.spike_stats.clear()
        logger.debug("SNN state and statistics reset.")

    def _init_weights(self) -> None:
        """重みの初期化。"""
        if isinstance(self.model, nn.Module):
            for m in self.model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)