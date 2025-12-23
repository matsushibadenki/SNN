# /snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (シグネチャ完全整合版)
# 目的: BitSpikeMambaが要求する vocab_size を含む全引数を網羅し、統合テストのエラーを解消する。

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    [Fix] BitSpikeMambaの初期化に必要な 'vocab_size' 引数不足を解消。
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
        
        # モデル構築用のパラメータ抽出
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        
        # ◾️ 修正箇所: 必須引数 'vocab_size' を追加 (テスト環境の期待値に適合)
        model_params = {
            "vocab_size": self.config_dict.get("vocab_size", 50257), # GPT-2デフォルト値をフォールバックに設定
            "d_model": self.config_dict.get("d_model", 128),
            "d_state": self.config_dict.get("d_state", 16),
            "d_conv": self.config_dict.get("d_conv", 4),
            "expand": self.config_dict.get("expand", 2),
            "num_layers": self.config_dict.get("num_layers", 4),
            "time_steps": self.config_dict.get("time_steps", 16),
            "neuron_config": self.config_dict.get("neuron_config", {"type": "lif"})
        }
        
        # キーワード引数として一括展開してモデルを初期化
        try:
            self.model = BitSpikeMamba(**model_params)
            logger.info(f"🧠 BitSpikeMamba model initialized with vocab_size={model_params['vocab_size']}")
        except TypeError as e:
            logger.error(f"❌ Initialization failed. Argument mismatch: {e}")
            raise e
        
        # 指定されたデバイスへ転送
        self.to(device)
        
        # チェックポイントの自動ロード (以前の修正を継承)
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # 形状不一致を許容する安全なロード
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.load_state_dict_safe(state_dict)
            except Exception as e:
                logger.error(f"⚠️ Checkpoint load failed: {e}")

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        """形状が一致する重みのみを適用する。"""
        model_dict = self.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Safe load: {len(filtered_dict)} consistent keys applied.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """SNNの期待する3D形状への変換と推論。"""
        if x.dim() == 2:
            # (Batch, SeqLen) -> トークンID入力の場合はそのまま、特徴量なら次元拡張
            if x.dtype == torch.long:
                pass # モデル内部のEmbedding層で処理
            else:
                x = x.unsqueeze(1).repeat(1, self.config_dict.get("time_steps", 16), 1)
        
        x = x.to(self.device)
        return self.model(x, **kwargs)

# エイリアス
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
