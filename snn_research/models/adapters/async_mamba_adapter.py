# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (完全整合版)
# 目的: テストコードおよび統合環境でのインポートエラーを解消し、重みの安全なロードを保証する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    [Fix] クラス名をテストコードの期待値 'AsyncBitSpikeMambaAdapter' に修正。
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # モデル本体の構築 (BitSpikeMamba)
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        self.model = BitSpikeMamba(config)

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        """
        形状が一致する重みのみをロードする堅牢なメソッド。
        """
        model_dict = self.state_dict()
        filtered_dict = {}
        mismatch_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    mismatch_keys.append(f"{k} (Checkpoint:{v.shape} vs Model:{model_dict[k].shape})")
        
        if mismatch_keys:
            logger.warning(f"⚠️ Weight shape mismatch detected in {len(mismatch_keys)} keys. Skipping.")

        # strict=False で安全にロード
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Safe load completed. Consistent keys: {len(filtered_dict)}")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """モデルの推論。"""
        return self.model(x, **kwargs)

# エイリアスの設定（他のモジュールでの参照用）
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
