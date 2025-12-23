# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (パラメータ展開修正版)
# 目的: BitSpikeMambaモデルの引数要件を満たし、統合テストの初期化エラーを完全に解消する。

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    [Fix] モデル初期化時の引数ミスマッチを解消。
    """
    def __init__(
        self, 
        config: Any, 
        device: str = "cpu", 
        checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        # config が OmegaConf の場合は辞書に変換
        if hasattr(config, "to_container"):
            self.config_dict = config.to_container(recursive=True)
        else:
            self.config_dict = dict(config)
            
        self.device = device
        
        # ◾️ 修正箇所: BitSpikeMambaが個別の引数を要求するため、辞書を展開して渡す
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        
        # 必須パラメータの抽出 (不足時はデフォルト値を設定)
        model_params = {
            "d_model": self.config_dict.get("d_model", 128),
            "d_state": self.config_dict.get("d_state", 16),
            "d_conv": self.config_dict.get("d_conv", 4),
            "expand": self.config_dict.get("expand", 2),
            "num_layers": self.config_dict.get("num_layers", 4),
            "time_steps": self.config_dict.get("time_steps", 16),
            "neuron_config": self.config_dict.get("neuron_config", {"type": "lif"})
        }
        
        # キーワード引数として展開してモデルを構築
        self.model = BitSpikeMamba(**model_params)
        
        # 指定されたデバイスへ転送
        self.to(device)
        
        # チェックポイントの自動ロード
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.load_state_dict_safe(state_dict)
                logger.info(f"✅ Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        """形状が一致する重みのみをロードする。"""
        model_dict = self.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
        
        self.load_state_dict(filtered_dict, strict=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # 入力を適切な形状とデバイスに調整
        if x.dim() == 2: # (Batch, Features) -> (Batch, Time, Features)
            x = x.unsqueeze(1).repeat(1, self.config_dict.get("time_steps", 16), 1)
        
        x = x.to(self.device)
        return self.model(x, **kwargs)

# エイリアス
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
