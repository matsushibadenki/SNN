# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncMamba アダプター (堅牢版)
# 目的: 重みの形状不一致を検知し、実行時エラーを防止しつつモデルをロードする。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncMambaAdapter(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # モデル本体の構築 (BitSpikeMamba等)
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        self.model = BitSpikeMamba(config)

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        """
        [Fix] 形状が一致する重みのみをロードする。
        不一致がある場合はログを出力し、そのレイヤーをスキップする（ランダム初期化を維持）。
        """
        model_dict = self.state_dict()
        filtered_dict = {}
        mismatch_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    mismatch_keys.append(f"{k} (Expected {model_dict[k].shape}, got {v.shape})")
            else:
                # 不要なキー（Unexpected keys）は無視
                pass

        if mismatch_keys:
            logger.warning(f"⚠️ Weight shape mismatch in {len(mismatch_keys)} keys. Skipping these keys.")
            for msg in mismatch_keys[:3]: # 最初の3つだけ詳細表示
                logger.warning(f"   - {msg}")

        # 一致する重みのみ適用 (strict=False)
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Loaded {len(filtered_dict)} consistent weight tensors.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model(x, **kwargs)
