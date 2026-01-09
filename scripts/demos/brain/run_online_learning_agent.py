# ファイルパス: scripts/demos/brain/run_online_learning_agent.py
# 日本語タイトル: Brain v4.0 Online Synesthesia Learner (5-Senses)
# 目的: チャットを通じて、視覚・触覚・嗅覚などの感覚入力をリアルタイムに「教育」できるエージェント。
#       ユーザーが「それはXXだよ」と教えると、即座にバックプロパゲーションが走り、概念を獲得する。

from snn_research.models.experimental.brain_v4 import SynestheticBrain
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))


class OnlineAgent:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        print(
            f"🧠 Initializing Online Learning Brain on {self.device.upper()}...")

        # 1. Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 5感対応Brainの初期化
        self.brain = SynestheticBrain(
            vocab_size=len(self.tokenizer),
            d_model=256,
            time_steps=8,
            tactile_dim=64,   # 触覚入力次元
            olfactory_dim=32,  # 嗅覚入力次元
            device=self.device
        ).to(self.device)

        # 2. オンライン学習用オプティマイザ
        # 学習率は少し高めに設定（即時適応のため）
        self.optimizer = optim.AdamW(self.brain.parameters(), lr=2e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.brain.eval()  # 基本は推論モード
        print("✅ Ready. I have generic senses but no knowledge yet.")

    def _generate_mock_sensory_data(self, modality: str, pattern_name: str):
        """
        デモ用に、コマンドに応じた模擬的な感覚信号（テンソル）を生成する。
        実際にはセンサーからの生データが入る場所。
        """
        tensor = None
        desc = ""

        if modality == 'vision':
            # 映像: (1, 8, 1, 28, 28)
            tensor = torch.zeros((1, 8, 1, 28, 28)).to(self.device)
            if "fall" in pattern_name:  # 落下
                desc = "[Vision: Object Falling ↓]"
                for t in range(8):
                    y = min(2 + t*3, 27)
                    tensor[0, t, 0, y:y+3, 12:15] = 1.0
            elif "fire" in pattern_name:  # 火 (揺らぎ)
                desc = "[Vision: Flickering Red/Orange 🔥]"
                for t in range(8):
                    tensor[0, t, 0, 14:, 10:18] = torch.rand(
                        (14, 8)).to(self.device)
            else:
                desc = f"[Vision: Unknown Static Pattern '{pattern_name}']"
                tensor = torch.rand((1, 8, 1, 28, 28)).to(self.device)

        elif modality == 'tactile':
            # 触覚: (1, 64) -> 時間展開される
            tensor = torch.zeros((1, 64)).to(self.device)
            if "hot" in pattern_name:
                desc = "[Tactile: High Temperature/Pain ♨️]"
                tensor[:] = 1.0  # 全ニューロン発火（激痛/熱）
            elif "soft" in pattern_name:
                desc = "[Tactile: Soft Texture ☁️]"
                tensor[0:10] = 0.5  # 一部が優しく発火
            else:
                desc = f"[Tactile: Random Sensation '{pattern_name}']"
                tensor = torch.rand((1, 64)).to(self.device)

        elif modality == 'olfactory':
            # 嗅覚: (1, 32)
            tensor = torch.zeros((1, 32)).to(self.device)
            if "sweet" in pattern_name or "flower" in pattern_name:
                desc = "[Olfactory: Sweet Floral Scent 🌸]"
                tensor[0, 5:10] = 1.0  # 特定の受容体パターン
            elif "gas" in pattern_name:
                desc = "[Olfactory: Pungent Chemical ⚠️]"
                tensor[0, 20:25] = 1.0
            else:
                desc = f"[Olfactory: Unknown Smell '{pattern_name}']"
                tensor = torch.rand((1, 32)).to(self.device)

        return tensor, desc

    def learn_concept(self, text_context: str, target_concept: str,
                      img_in=None, tac_in=None, olf_in=None, epochs=5):
        """
        即座に学習を行うメソッド (Short-term Plasticity)
        """
        print(
            f"\n   🎓 Learning that this sensation means '{target_concept}'...")
        self.brain.train()

        # コンテキストと正解ラベルの準備
        inputs = self.tokenizer(
            text_context, return_tensors="pt").input_ids.to(self.device)
        target_id = self.tokenizer.encode(
            " " + target_concept, return_tensors="pt")[0, 0].to(self.device)
        target_tensor = torch.tensor([target_id], device=self.device)

        for i in range(epochs):
            self.optimizer.zero_grad()
            logits = self.brain(
                text_input=inputs,
                image_input=img_in,
                tactile_input=tac_in,
                olfactory_input=olf_in
            )
            # 次の単語予測のロス
            loss = self.criterion(logits[:, -1, :], target_tensor)
            loss.backward()
            self.optimizer.step()

            if i == 0 or i == epochs-1:
                print(f"     Iter {i+1}/{epochs}: Loss={loss.item():.4f}")

        self.brain.eval()
        print("   ✅ Learned.\n")

    def chat_loop(self):
        print("\n=======================================================")
        print(" 🧠 Brain v4.0 Interactive Synesthesia Agent")
        print("=======================================================")
        print(" Commands:")
        print("   /see [falling|fire]   : Simulate Vision")
        print("   /touch [hot|soft]     : Simulate Tactile")
        print("   /smell [flower|gas]   : Simulate Olfactory")
        print(
            "   /teach [concept]      : Teach the brain what the sensation is")
        print("   (Normal text)         : Chat with the brain")
        print("-------------------------------------------------------\n")

        # 状態保持用
        current_sensation = {"img": None, "tac": None, "olf": None}

        while True:
            try:
                user_input = input("User: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue

                # --- 感覚入力コマンドの処理 ---
                if user_input.startswith("/"):
                    cmd_parts = user_input.split()
                    cmd = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else "unknown"

                    if cmd == "/see":
                        tensor, desc = self._generate_mock_sensory_data(
                            'vision', arg)
                        current_sensation["img"] = tensor

                        print(f"   {desc}")

                    elif cmd == "/touch":
                        tensor, desc = self._generate_mock_sensory_data(
                            'tactile', arg)
                        current_sensation["tac"] = tensor

                        print(f"   {desc}")

                    elif cmd == "/smell":
                        tensor, desc = self._generate_mock_sensory_data(
                            'olfactory', arg)
                        current_sensation["olf"] = tensor

                        print(f"   {desc}")

                    elif cmd == "/teach":
                        # 直前の感覚入力に対してラベル付けを行う
                        target = arg
                        prompt = "I feel this sensation. It is"
                        self.learn_concept(prompt, target,
                                           current_sensation["img"],
                                           current_sensation["tac"],
                                           current_sensation["olf"])
                        # 感覚リセット
                        current_sensation = {
                            "img": None, "tac": None, "olf": None}

                        continue

                    else:
                        print("   ⚠️ Unknown command.")
                        continue

                    # 感覚入力後の脳の反応を見る
                    print("   Brain is sensing... what does it think?")
                    prompt = "I feel this sensation. It is"
                    # 入力がないモダリティはNoneのまま渡す
                    with torch.no_grad():
                        logits = self.brain(
                            text_input=self.tokenizer(
                                prompt, return_tensors="pt").input_ids.to(self.device),
                            image_input=current_sensation["img"],
                            tactile_input=current_sensation["tac"],
                            olfactory_input=current_sensation["olf"]
                        )
                        pred_id = torch.argmax(logits[:, -1, :]).item()
                        pred_word = self.tokenizer.decode([pred_id]).strip()
                        print(f"   Brain: \"{pred_word}\"")
                        print(
                            "   (If wrong, type '/teach [correct_word]' to learn)")

                else:
                    # --- 通常会話 ---
                    # 脳にチャットさせる（感覚入力が残っていればそれも付与）
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            f"User: {user_input}\nBrain:", return_tensors="pt").input_ids.to(self.device)
                        logits = self.brain(
                            text_input=inputs,
                            # コンテキストとして感覚を保持
                            image_input=current_sensation["img"],
                            tactile_input=current_sensation["tac"],
                            olfactory_input=current_sensation["olf"]
                        )
                        pred_id = torch.argmax(logits[:, -1, :]).item()
                        pred_word = self.tokenizer.decode([pred_id]).strip()
                        print(f"Brain: {pred_word} ...")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    agent = OnlineAgent()
    agent.chat_loop()
