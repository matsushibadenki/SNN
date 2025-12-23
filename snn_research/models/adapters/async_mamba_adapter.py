# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (最終整合版)
# 目的: テストコードからの 'device' 引数による初期化に対応し、統合テストのエラーを解消する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    [Fix] テストコード(test_brain_integration.py)が期待する 'device' 引数に対応。
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        
        # モデル本体の構築 (BitSpikeMamba)
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        self.model = BitSpikeMamba(config)
        
        # 指定されたデバイスへ転送
        self.to(device)
        logger.info(f"🚀 AsyncBitSpikeMambaAdapter initialized on device: {device}")

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        """
        形状が一致する重みのみをロードする。不一致がある場合はログを出力してスキップ。
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
            logger.warning(f"⚠️ Weight shape mismatch detected in {len(mismatch_keys)} keys. Consistent weights will be loaded.")

        # 一致した重みのみ適用
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Safe load completed. Consistent weight tensors: {len(filtered_dict)}")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """モデルの推論実行。"""
        # 入力をモデルと同じデバイスへ転送
        x = x.to(self.device)
        return self.model(x, **kwargs)

# エイリアスの設定（後方互換性用）
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
