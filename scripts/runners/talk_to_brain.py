# ファイルパス: scripts/runners/talk_to_brain.py
# 日本語タイトル: Talk to Brain v20 (Interactive Mode)
# 目的・内容:
#   学習済みBitSpikeMambaモデルと直接対話するCLIツール。
#   AsyncArtificialBrainを経由せず、同期的に推論を行うことで応答を確実に確認する。

import sys
import os
import torch
import logging

# プロジェクトルートへのパス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ログ設定（シンプルに）
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TalkToBrain")

def main():
    print("\n🧠 Awakening Brain v20... (Loading Weights)\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 学習時と同じ構成
    mamba_config = {
        "d_model": 128,
        "d_state": 32,
        "num_layers": 4,
        "tokenizer": "gpt2"
    }
    
    # アダプター初期化
    brain_adapter = AsyncBitSpikeMambaAdapter(mamba_config, device=device)
    
    if brain_adapter.model is None:
        print("❌ Failed to load model. Please check logs.")
        return

    print("\n✨ Brain is Ready! (Type 'exit' to quit)\n")
    print("--------------------------------------------------")

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("🧠 Brain: Farewell, my friend.")
                break
            
            if not user_input.strip():
                continue

            # 推論実行 (同期的に呼び出し)
            print("   (Thinking...)")
            response = brain_adapter.process(user_input)
            
            # 結果表示
            print(f"🧠 Brain: {response}")
            print("--------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\n🧠 Brain: Interrupted. Sleep mode activated.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()