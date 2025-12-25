# ファイルパス: scripts/trainers/train_brain_v5_distinct.py
# 日本語タイトル: Brain v2.0 Distinct Trainer (Spartan Mode)
# 目的・内容: 
#   モード崩壊（金太郎飴状態）を打破し、質問ごとに明確に異なる回答ができるまで徹底学習させる。
#   Target Loss: 0.05 (完全暗記)
#   [Fix] mypyエラー修正: layer.set_stateful呼び出し時の型キャストを追加。

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Any, cast, Tuple # 追加

# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.core.mamba_core import SpikingMambaBlock
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
from spikingjelly.activation_based import functional, surrogate, base

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("BrainTrainer")

# --- 1. 定義 ---
class SimpleLIFNeuron(base.MemoryModule):
    def __init__(self, features, tau_mem=4.0, base_threshold=0.01): # tau_memを10.0->4.0に短縮
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
    # [Fix] Aligned signature with parent (input_ids instead of x)
    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        functional.reset_net(self)
        
        # Use embedding based on input type check (as done in parent)
        # But this script assumes input_ids is LongTensor
        x_embed = self.embedding(input_ids)
        x_out = x_embed
        
        # [Fix] キャストして mypy エラー "Tensor not callable" を回避
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                 cast(Any, layer).set_stateful(True)
        
        for _ in range(self.time_steps):
            x = x_embed 
            for layer in self.layers:
                x = layer(x)
            x_out = x
            
        # [Fix] キャストして mypy エラー回避
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                 cast(Any, layer).set_stateful(False)
        
        # Output layers are assumed to be nn.Module, no cast needed if inferred correctly, 
        # but parent fix used Any cast.
        logits = self.output_projection(self.norm(x_out))
        
        # Return tuple matching signature
        device = input_ids.device
        return logits, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

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
            # tau_mem = 4.0 に設定
            layer.lif_conv = SimpleLIFNeuron(features=feat_conv, tau_mem=4.0, base_threshold=0.01).to(device)
            layer.lif_out = SimpleLIFNeuron(features=feat_out, tau_mem=4.0, base_threshold=0.01).to(device)

def generate_response(model, tokenizer, device, prompt):
    model.eval()
    functional.reset_net(model)
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids
        for _ in range(15):
            functional.reset_net(model)
            # forward returns tuple
            logits, _, _ = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# --- 2. Dataset ---
class DistinctDataset(Dataset):
    def __init__(self, tokenizer, block_size=64):
        self.tokenizer = tokenizer
        self.block_size = block_size
        corpus = [
            "User: Hello. AI: Hi there.",
            "User: Who are you? AI: I am Brain v2.0.",
            "User: What is SNN? AI: Spiking Neural Network.", # 短くシンプルに
            "User: I see an apple. AI: An apple is red.",
            "User: Do you dream? AI: Yes, I dream.",
            "User: How are you? AI: I am normal.",
        ]
        self.data = []
        # 反復回数を調整 (100回)
        for _ in range(100): 
            for text in corpus:
                ids = tokenizer.encode(text, add_special_tokens=False)
                self.data.extend(ids)
                self.data.append(tokenizer.eos_token_id)

    def __len__(self): return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        if start + self.block_size + 1 > len(self.data): start = len(self.data) - self.block_size - 1
        chunk = torch.tensor(self.data[start : start + self.block_size + 1], dtype=torch.long)
        return chunk[:-1], chunk[1:]

# --- 3. Trainer ---
def train():
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-3
    TARGET_LOSS = 0.05 # ★鬼の目標値
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    logger.info(f"🚀 Starting Distinct Training on {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = FixedBitSpikeMamba(
        vocab_size=tokenizer.vocab_size, d_model=128, d_state=32, d_conv=4, expand=2, num_layers=4, time_steps=4,
        neuron_config={"type": "lif"} 
    ).to(device)
    
    force_replace_components(model, device)

    dataset = DistinctDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    logger.info(">>> Training Loop Started...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            functional.reset_net(model)
            # forward returns tuple
            logits, _, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
        if (epoch+1) % 5 == 0:
            # ★2つの異なる質問をして、答えが使い分けられているか確認
            r1 = generate_response(model, tokenizer, device, "User: Hello. AI:")
            r2 = generate_response(model, tokenizer, device, "User: Who are you? AI:")
            print(f"   [Test 1] {r1}")
            print(f"   [Test 2] {r2}")
            
        if avg_loss < TARGET_LOSS:
            logger.info(f"🎯 Perfect Convergence (Loss < {TARGET_LOSS}). Stopping.")
            break
        
        torch.save(model.state_dict(), "models/checkpoints/trained_brain_v20.pth")

    logger.info("🎉 Done.")

if __name__ == "__main__":
    train()
