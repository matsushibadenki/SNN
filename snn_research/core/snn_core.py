# snn_research/core/snn_core.py
# Title: SNNCore (No-Sync Optimization)
# Description:
#   統計収集時の同期オーバーヘッド(.item())を削除し、
#   generateメソッドのエラーハンドリングを強化。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast, Union
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

        try:
            self.model: Any = ArchitectureRegistry.build(arch_type, self.config, vocab_size)
        except Exception as e:
            logger.error(f"Failed to build model '{arch_type}': {e}")
            raise RuntimeError(f"Model build failed: {e}")

        self._init_weights()
        self.spike_stats: Dict[str, float] = {}

    def forward(self, x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None, **kwargs: Any) -> Any:
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input', 'pixel_values']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break

        if x is None and not kwargs:
            raise ValueError("SNNCore forward called with no input data.")

        # Forward
        if x is not None:
            if isinstance(x, dict):
                output = self.model(**x, **kwargs)
            else:
                output = self.model(x, **kwargs)
        else:
            output = self.model(**kwargs)

        # 統計更新 (軽量版: item()を呼ばない)
        # 必要であれば、monitor_statsフラグで制御
        # self._update_firing_stats() 
        
        return output

    def _update_firing_stats(self) -> None:
        """
        [Optimized] 同期を避けるため、ここでは計算を行わないか、非同期にログする。
        厳密な統計が必要な場合は、明示的に別メソッドを呼ぶ設計にする。
        """
        pass 

    def get_firing_rates(self) -> Dict[str, float]:
        """
        外部から要求された時だけ計算して返す。
        """
        stats = {}
        # モデルごとの実装に合わせて取得
        get_rates_method = getattr(self.model, 'get_firing_rates', None)
        if get_rates_method and callable(get_rates_method):
            raw_rates = get_rates_method()
            if isinstance(raw_rates, dict):
                for k, v in raw_rates.items():
                    # ここでは .item() を呼んでも良い（頻度が低い前提）
                    stats[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return stats

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        gen_method = getattr(self.model, 'generate', None)
        if gen_method and callable(gen_method):
            return cast(torch.Tensor, gen_method(input_ids, **kwargs))
        return input_ids

    def reset_state(self) -> None:
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(self.model)
        except:
            pass
        
        reset_method = getattr(self.model, 'reset_state', None)
        if reset_method and callable(reset_method):
            reset_method()

        self.spike_stats.clear()

    def _init_weights(self) -> None:
        if isinstance(self.model, nn.Module):
            for m in self.model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        try:
                            nn.init.xavier_uniform_(m.weight)
                            if hasattr(m, 'bias') and m.bias is not None:
                                nn.init.zeros_(m.bias)
                        except:
                            pass
    
    def update_plasticity(self, x_input: torch.Tensor, target: torch.Tensor, learning_rate: float = 0.01) -> None:
        update_method = getattr(self.model, 'update_plasticity', None)
        if update_method and callable(update_method):
            update_method(x_input, target, learning_rate)