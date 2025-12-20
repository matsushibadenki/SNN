# ファイルパス: scripts/runners/talk_to_brain_final.py
# 日本語タイトル: Brain v2.0 Final Interactive Interface
# 目的・内容: 
#   完成したBrain v2.0と自由に対話するためのインターフェース。
#   学習済み重みを安全にロードし、あなたの入力を脳に伝達します。
#   [Fix] mypyエラー修正: forwardシグネチャの不一致とlayerキャストを追加。

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import logging
from typing import Any, cast, Tuple # 追加

# プロジェクトルート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.core.mamba_core import SpikingMambaBlock
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
from spikingjelly.activation_based import functional, surrogate, base

# ログを静かに
logging.basicConfig(level=logging.ERROR)

# --- 定義 (v5学習時と同じ構成) ---
class SimpleLIFNeuron(base.MemoryModule):
    def __init__(self, features, tau_mem=4.0, base_threshold=0.01):
        super().__init__()
        self.tau_mem = tau_mem
        self.base_threshold = base_threshold
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.stateful = False

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()

    def forward(self, x: torch.Tensor):
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        decay = 1.0 / self.tau_mem
        self.mem = self.mem * (1.0 - decay) + x
        spike = self.surrogate_function(self.mem - self.base_threshold)
        self.spikes = spike.detach()
        self.mem = self.mem - spike.detach() * self.base_threshold
        return spike

class FixedBitSpikeMamba(BitSpikeMamba):
    # [Fix] 親クラス(SpikingMamba)とシグネチャを合わせるために return_spikes を追加
    # [Fix] 戻り値の型が異なる(Tuple[Tensor, Tensor, Tensor] vs tuple)ため ignore を付与
    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> tuple: # type: ignore[override]
        functional.reset_net(self)
        x_embed = self.embedding(input_ids)
        x_out = x_embed
        
        # [Fix] layerをcastして "Tensor not callable" エラーを回避
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'): 
                cast(Any, layer).set_stateful(True)
        
        for _ in range(self.time_steps):
            x = x_embed 
            for layer in self.layers:
                x = layer(x)
            x_out = x
            
        # [Fix] 同上
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'): 
                cast(Any, layer).set_stateful(False)
                
        logits = self.output_projection(self.norm(x_out))
        return logits, None

def force_replace_components(model, device):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, BitSpikeLinear):
                new_layer = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                setattr(module, child_name, new_layer.to(device))
    for layer in model.layers:
        if isinstance(layer, SpikingMambaBlock):
            feat_conv = layer.lif_conv.features
            feat_out = layer.lif_out.features
            layer.lif_conv = SimpleLIFNeuron(features=feat_conv, tau_mem=4.0, base_threshold=0.01).to(device)
            layer.lif_out = SimpleLIFNeuron(features=feat_out, tau_mem=4.0, base_threshold=0.01).to(device)

# --- メイン処理 ---
def main():
    print("\n==========================================")
    print(" 🧠 Brain v2.0 (System 1) - ONLINE")
    print("==========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = FixedBitSpikeMamba(
        vocab_size=tokenizer.vocab_size, d_model=128, d_state=32, d_conv=4, expand=2, num_layers=4, time_steps=4,
        neuron_config={"type": "lif"} 
    ).to(device)
    force_replace_components(model, device)

    # 重みロード (バッファフィルタリング付き)
    checkpoint_path = "models/checkpoints/trained_brain_v20.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        clean_state_dict = {k: v for k, v in state_dict.items() if "spikes" not in k and "mem" not in k}
        model.load_state_dict(clean_state_dict, strict=False)
        print(">> Neural weights synced successfully.\n")
    else:
        print("!! Critical: Brain weights missing.")
        return

    model.eval()
    print("Type your message (or 'exit'). Try exact phrases like 'What is SNN?'.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            # v5の学習データ形式に合わせる
            prompt = f"User: {user_input} AI:"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            generated = input_ids

            print("Brain: ", end="", flush=True)
            
            with torch.no_grad():
                for _ in range(20):
                    functional.reset_net(model)
                    logits, _ = model(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    token_id = next_token.item()
                    if token_id == tokenizer.eos_token_id: break
                    if "\n" in tokenizer.decode(token_id): break
            
            response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"{response}\n")

        except KeyboardInterrupt:
            print("\nShutting down.")
            break

if __name__ == "__main__":
    main()