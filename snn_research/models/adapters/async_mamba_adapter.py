# ファイルパス: snn_research/models/adapters/async_mamba_adapter.py
# 日本語タイトル: Async BitSpike Mamba Adapter (Fixed: Decode Error)
# 目的・内容:
#   BitSpikeMambaモデルのアダプター。
#   修正: tokenizer.decode に渡す前に Tensor を int に変換し、TypeError を回避。

import torch
import logging
import threading
import os
from typing import Any, Optional
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

logger = logging.getLogger(__name__)

class AsyncBitSpikeMambaAdapter:
    """
    AsyncArtificialBrain用のBitSpikeMambaラッパー (Thread-Safe & Emotion-Aware)。
    """
    def __init__(self, model_config: dict, device: str = "cpu", checkpoint_path: Optional[str] = "models/checkpoints/trained_brain_v20.pth"):
        self.device = device
        self.model_config = model_config
        self.model: Optional[BitSpikeMamba] = None
        self._lock = threading.Lock()
        
        # 感情状態 (デフォルト: 普通)
        self.mood_state = "Neutral"
        self.mood_prompt = ""
        
        logger.info(f"🔌 Initializing BitSpikeMamba Adapter on {device}...")
        
        # Tokenizer
        tokenizer_name = model_config.get("tokenizer", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"⚠️ Tokenizer load failed: {e}")
            self.tokenizer = None

        # Model Building
        try:
            self.model = BitSpikeMamba(
                vocab_size=self.tokenizer.vocab_size if self.tokenizer else 1000,
                d_model=model_config.get("d_model", 128),
                d_state=model_config.get("d_state", 32),
                d_conv=model_config.get("d_conv", 4),
                expand=model_config.get("expand", 2),
                num_layers=model_config.get("num_layers", 4),
                time_steps=model_config.get("time_steps", 4),
                neuron_config={"type": "lif", "tau_mem": 2.0}
            ).to(device)
            
            # checkpoint_path引数が指定されていればそれを使い、なければデフォルトパスを確認
            ckpt = checkpoint_path if checkpoint_path else "models/checkpoints/trained_brain_v20.pth"
            
            if os.path.exists(ckpt):
                logger.info(f"📂 Loading trained weights from {ckpt}...")
                state_dict = torch.load(ckpt, map_location=device)
                self.model.load_state_dict(state_dict)
                logger.info("🎉 Weights loaded successfully!")
            else:
                logger.warning(f"⚠️ Checkpoint not found at {ckpt}. Using random weights.")

            self.model.eval()
            logger.info("✅ BitSpikeMamba Model ready.")
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            self.model = None

    def update_mood(self, valence: float, arousal: float):
        """
        Amygdalaからの信号で機嫌を更新する。
        """
        with self._lock:
            if valence > 0.5:
                self.mood_state = "Happy"
                self.mood_prompt = " (Feeling: Happy and Excited) "
            elif valence < -0.5:
                self.mood_state = "Angry/Sad"
                self.mood_prompt = " (Feeling: Angry and Defensive) "
            else:
                self.mood_state = "Neutral"
                self.mood_prompt = ""
            
            logger.info(f"🧠 Brain Mood Updated: {self.mood_state} (V:{valence:.2f})")

    def process(self, input_payload: Any) -> Optional[str]:
        if not self.model or not self.tokenizer:
            return "Error: Model not ready."

        with self._lock:
            raw_text = str(input_payload)
            # Prompt Engineering: 学習データ形式に合わせる
            contextual_input = f"{self.mood_prompt}User: {raw_text} AI:"
            
            logger.debug(f"🧠 [Thinking] Input: {contextual_input}")

            try:
                input_ids = self.tokenizer(contextual_input, return_tensors="pt").to(self.device).input_ids
                
                # Generation Settings
                max_new_tokens = 30 # 長すぎると崩れるので短めに
                temperature = 0.6   # 決定論的に寄せる
                
                generated_ids = input_ids
                
                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        # ★重要: 前回の推論で蓄積した膜電位をリセットする★
                        functional.reset_net(self.model)
                        
                        # Forward pass
                        logits, _, _ = self.model(generated_ids)
                        
                        # Next token prediction
                        next_token_logits = logits[:, -1, :] / temperature
                        
                        # Repetition Penalty (簡易版)
                        for token_id in set(generated_ids[0].tolist()):
                            next_token_logits[0, token_id] /= 1.1

                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        
                        # EOSトークンまたは改行が来たら終了
                        # ★修正: .item() で整数値を取り出してから比較/デコードする
                        token_id_int = next_token.item()
                        
                        if token_id_int == self.tokenizer.eos_token_id:
                            break
                        
                        # 学習データが1行会話なので、改行でも切る
                        if "\n" in self.tokenizer.decode(token_id_int):
                            break
                
                # Output Decoding
                new_tokens = generated_ids[0, input_ids.shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                logger.info(f"🗣️ Brain Says ({self.mood_state}): '{response}'")
                return response
                
            except Exception as e:
                logger.error(f"💥 Inference Error: {e}", exc_info=True)
                return "..."