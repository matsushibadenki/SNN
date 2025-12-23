# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (完全統合版)
# 目的: テストコードからの 'checkpoint_path' 引数に対応し、自動ロード機能を含む初期化を実現する。

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    [Fix] テストコード(test_brain_integration.py)が期待する 'checkpoint_path' 引数に対応し、
    初期化時に重みを安全に自動ロードする機能を追加。
    """
    def __init__(
        self, 
        config: Dict[str, Any], 
        device: str = "cpu", 
        checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # 1. モデル本体の構築 (BitSpikeMamba)
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        self.model = BitSpikeMamba(config)
        
        # 2. 指定されたデバイスへ転送
        self.to(device)
        
        # 3. [Fix] チェックポイントが指定されている場合は自動ロードを実行
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.load_state_dict_safe(state_dict)
                logger.info(f"✅ Automatically loaded checkpoint from: {checkpoint_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint in __init__: {e}")
        elif checkpoint_path:
            logger.warning(f"⚠️ Checkpoint path provided but not found: {checkpoint_path}")

        logger.info(f"🚀 AsyncBitSpikeMambaAdapter initialized on device: {device}")

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
            logger.warning(f"⚠️ Weight shape mismatch detected in {len(mismatch_keys)} keys. Consistent weights will be loaded.")

        # strict=False で安全にロードを実行
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Safe load completed. Consistent weight tensors: {len(filtered_dict)}")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """モデルの推論実行。"""
        # 入力をモデルと同じデバイスへ転送
        x = x.to(self.device)
        return self.model(x, **kwargs)

# エイリアスの設定（後方互換性および既存コード用）
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
