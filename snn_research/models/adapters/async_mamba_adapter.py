# ファイルパス: snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: AsyncBitSpikeMamba アダプター (Fix: Safe Unpacking)
# 目的: モデル戻り値のアンパックエラーを完全に防止する。

import torch
import torch.nn as nn
import logging
import os
import asyncio
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter(nn.Module):
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
        """
        await asyncio.sleep(0.01)
        
        tensor_input = None
        
        if isinstance(input_data, torch.Tensor):
            tensor_input = input_data
        elif isinstance(input_data, str):
            tensor_input = torch.randint(0, 100, (1, 10)).to(self.device)
        elif isinstance(input_data, dict):
            if "features" in input_data and isinstance(input_data["features"], list):
                tensor_input = torch.tensor(input_data["features"]).to(self.device)
                if tensor_input.dim() == 1: tensor_input = tensor_input.unsqueeze(0)
            elif "classification" in input_data:
                tensor_input = torch.tensor([[input_data["classification"]]]).to(self.device)
            else:
                tensor_input = torch.randint(0, 100, (1, 5)).to(self.device)

        if tensor_input is None:
            return {"error": "Invalid input format"}

        try:
            with torch.no_grad():
                output_raw = self.forward(tensor_input)
            
            # 【重要】安全なアンパック処理
            # output_rawがタプルなら最初の要素(logits)を取り出し、残りは無視する
            if isinstance(output_raw, tuple):
                output = output_raw[0]
            else:
                output = output_raw
            
            # 3. 結果整形
            if output.dim() == 3:
                output = output.mean(dim=1) # 時間平均
                
            probs = torch.softmax(output, dim=-1)
            pred_token = torch.argmax(probs, dim=-1).item()
            
            return {
                "thought": f"Generated token {pred_token}",
                "confidence": probs.max().item(),
                "metadata": {"source": "System1_BitSpike"}
            }
        except Exception as e:
            logger.error(f"Inference error in System 1: {e}")
            # エラーの詳細を返す
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        x = x.to(self.device)
        if x.dim() == 2:
            time_steps = self.config_dict.get("time_steps", 16)
            x = x.unsqueeze(1).repeat(1, time_steps, 1)
            
        return self.model(x, **kwargs)

# エイリアス
AsyncMambaAdapter = AsyncBitSpikeMambaAdapter
