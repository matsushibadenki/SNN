# ファイルパス: snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (Fix: Dict Input Support)
# 目的: Brain Kernelからの辞書型入力(Visual Cortex出力等)に対応し、エラーを回避する。

import torch
import torch.nn as nn
import logging
import os
import asyncio
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行環境とBitSpikeMambaモデルを橋渡しするアダプター。
    """
    def __init__(
        self, 
        config: Any, 
        device: str = "cpu", 
        checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        if hasattr(config, "to_container"):
            self.config_dict = config.to_container(recursive=True)
        else:
            self.config_dict = dict(config)
            
        self.device = device
        
        # BitSpikeMambaの初期化
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        
        model_params = {
            "vocab_size": self.config_dict.get("vocab_size", 50257),
            "d_model": self.config_dict.get("d_model", 128),
            "d_state": self.config_dict.get("d_state", 16),
            "d_conv": self.config_dict.get("d_conv", 4),
            "expand": self.config_dict.get("expand", 2),
            "num_layers": self.config_dict.get("num_layers", 4),
            "time_steps": self.config_dict.get("time_steps", 16),
            "neuron_config": self.config_dict.get("neuron_config", {"type": "lif"})
        }
        
        try:
            self.model = BitSpikeMamba(**model_params)
            logger.info(f"🧠 BitSpikeMamba model initialized with vocab_size={model_params['vocab_size']}")
        except TypeError as e:
            logger.error(f"❌ Initialization failed. Argument mismatch: {e}")
            raise e
        
        self.to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.load_state_dict_safe(state_dict)
            except Exception as e:
                logger.error(f"⚠️ Checkpoint load failed: {e}")

    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]):
        model_dict = self.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        self.load_state_dict(filtered_dict, strict=False)
        logger.info(f"✅ Safe load: {len(filtered_dict)} consistent keys applied.")

    async def process(self, input_data: Union[torch.Tensor, str, Dict[str, Any]]) -> Any:
        """
        Brain Kernelから呼び出される非同期処理のエントリーポイント。
        入力の型に応じて前処理を行い、推論を実行する。
        """
        # シミュレーション用遅延
        await asyncio.sleep(0.01)
        
        tensor_input = None
        
        # 1. 入力の型判定と変換
        if isinstance(input_data, torch.Tensor):
            tensor_input = input_data
        
        elif isinstance(input_data, str):
            # 文字列の場合はトークナイズが必要だが、ここでは簡易的にダミーテンソル化
            # 本来はTokenizerを使用する
            # logger.info(f"🗣️ Processing text input: {input_data}")
            tensor_input = torch.randint(0, 100, (1, 10)).to(self.device) # Dummy token ids
            
        elif isinstance(input_data, dict):
            # 視覚野などからのメタデータ付き入力
            # logger.info(f"🧠 Processing multimodal input: {input_data.keys()}")
            
            # 特徴量が来ていればそれを使う、なければ分類結果などをテキスト化して処理
            if "features" in input_data and isinstance(input_data["features"], list):
                # 特徴量をTensorに戻す (リスト経由の場合)
                tensor_input = torch.tensor(input_data["features"]).to(self.device)
                if tensor_input.dim() == 1: tensor_input = tensor_input.unsqueeze(0)
            elif "classification" in input_data:
                # 分類IDをトークンIDとして扱う簡易実装
                cls_id = input_data["classification"]
                tensor_input = torch.tensor([[cls_id]]).to(self.device)
            else:
                # 何もなければダミー
                tensor_input = torch.randint(0, 100, (1, 5)).to(self.device)

        if tensor_input is None:
            return {"error": "Invalid input format"}

        # 2. 推論実行 (同期メソッド呼び出し)
        with torch.no_grad():
            output = self.forward(tensor_input)
            
        # 3. 結果整形 (Logits -> 次のトークン予測など)
        # SNN出力は (Batch, Time, Vocab) または (Batch, Vocab)
        if output.dim() == 3:
            output = output.mean(dim=1) # 時間平均
            
        probs = torch.softmax(output, dim=-1)
        pred_token = torch.argmax(probs, dim=-1).item()
        
        return {
            "thought": f"Generated token {pred_token}",
            "confidence": probs.max().item(),
            "metadata": {"source": "System1_BitSpike"}
        }

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """PyTorch標準のForward"""
        x = x.to(self.device)
        
        # 次元調整: (Batch, Seq) -> (Batch, Time, Seq) or similar depends on model
        # BitSpikeMambaが時間次元をどう扱うかに依存。
        # ここでは入力が (Batch, Seq) または (Batch, Dim) と仮定し、時間方向に拡張
        if x.dim() == 2:
            time_steps = self.config_dict.get("time_steps", 16)
            # Token ID (Long) なら Embedding層で処理されるのでそのまま渡す場合もあるが
            # SNNの場合は入力を時間的に繰り返すのが一般的
            x = x.unsqueeze(1).repeat(1, time_steps, 1)
            
        return self.model(x, **kwargs)

# エイリアス
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
