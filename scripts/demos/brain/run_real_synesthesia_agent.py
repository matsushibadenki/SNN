# ファイルパス: scripts/demos/brain/run_real_synesthesia_agent.py
# 日本語タイトル: Brain v4.0 Real Synesthesia Agent (Mypy Fixed)
# 目的: 学習済みの共感覚回路(Brain v4)とReasoningEngineを統合し、
#       「実際に画像を見て」会話できるエージェントを実現する。

from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.utils.generative_adapter import SNNGenerativeAdapter
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.models.experimental.brain_v4 import SynestheticBrain
import sys
import os
import torch
import logging
from PIL import Image
from torchvision import transforms, datasets
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))

# ★修正: パスを固定


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("RealBrain")


class SynestheticAgent:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        print(f"Initializing Brain v4.0 on {self.device.upper()}...")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Brain v4.0 (共感覚脳) のロード
        self.brain = SynestheticBrain(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256, num_layers=4, time_steps=4,
            device=self.device
        ).to(self.device)

        # 重みのロード (共感覚回路)
        ckpt_path = "models/checkpoints/brain_v4_synesthesia.pth"
        if os.path.exists(ckpt_path):
            self.brain.load_state_dict(torch.load(
                ckpt_path, map_location=self.device), strict=False)
            print("✅ Synesthetic Circuits (Vision-Language) Loaded.")
        else:
            print("⚠️ Warning: Synesthesia checkpoint not found. Vision will be random.")

        # 3. 推論エンジンの構築
        self.gen_adapter = SNNGenerativeAdapter(
            self.brain.core_brain, device=self.device)
        self.astrocyte = AstrocyteNetwork(
            initial_energy=1000.0, max_energy=1000.0, recovery_rate=10.0)
        self.rag = RAGSystem(embedding_dim=256)

        self.rag.add_knowledge(
            "Brain v4.0 possesses artificial synesthesia, allowing it to see and feel concepts.")

        self.engine = ReasoningEngine(
            generative_model=self.gen_adapter,
            astrocyte=self.astrocyte,
            tokenizer=self.tokenizer,
            rag_system=self.rag,
            d_model=256,
            num_thinking_paths=1,
            max_thinking_steps=64,
            sandbox_timeout=10.0,
            device=self.device
        )

        # 画像前処理 (MNIST学習時と同じ正規化)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def see_image(self, image_path: str) -> str:
        """本物の視覚野を使って画像を見る"""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file not found."

            # 画像読み込みと前処理
            img = Image.open(image_path)
            img_tensor = self.transform(
                img).view(-1).to(self.device)  # Flatten [784]

            start_token = self.tokenizer.encode(
                "Digit", return_tensors="pt")[0, 0].item()

            gen_ids = self.brain.generate(
                img_tensor.unsqueeze(0),  # Batch dim
                start_token_id=start_token,
                max_new_tokens=20
            )

            description = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True)
            description = description.replace("Digitit", "Digit").strip()
            return f"I see: {description}"

        except Exception as e:
            return f"Vision Error: {e}"

    def generate_test_image(self, digit: int):
        """テスト用にMNISTから画像を生成して保存する"""
        try:
            mnist = datasets.MNIST('./data', train=False, download=True)
            for img, label in mnist:
                if label == digit:
                    filename = f"test_digit_{digit}.png"
                    img.save(filename)
                    print(
                        f"   [🖼️ Image Created] Saved '{filename}'. You can see it with '/image {filename}'")
                    return
        except Exception as e:
            print(f"Error generating image: {e}")

    def chat(self, user_input: str):
        # コマンド処理
        if user_input.startswith("/image"):
            path = user_input.replace("/image", "").strip()
            print(f"   [👁️ Synesthetic Vision] Focusing on {path}...")
            vision_result = self.see_image(path)
            print(f"   [🧠 Visual Cortex] {vision_result}")

            prompt = f"User: I am showing you an image. {vision_result} What do you feel?\nBrain:"

        elif user_input.startswith("/generate"):
            try:
                digit = int(user_input.split()[-1])
                self.generate_test_image(digit)
                return
            except Exception:
                print("Usage: /generate [0-9]")
                return
        else:
            prompt = f"User: {user_input}\nBrain:"

        result = self.engine.process(prompt)

        if "error" in result:
            print(f"Brain Error: {result['error']}")
        else:
            final_text = result.get(
                "final_text", "").replace(prompt, "").strip()
            print(f"Brain: {final_text}\n")


def main():
    agent = SynestheticAgent()
    print("\n=======================================================")
    print(" 🧠 Brain v4.0 Real Synesthesia Agent")
    print("=======================================================\n")
    print("Commands:")
    print("  - Chat:        'Hello', 'What is SNN?'")
    print("  - Create Img:  '/generate 7' (Creates test_digit_7.png)")
    print("  - See Image:   '/image test_digit_7.png'")
    print("-------------------------------------------------------\n")

    while True:
        try:
            u = input("User: ")
            if u.lower() in ["exit", "quit"]:
                break
            if not u.strip():
                continue
            agent.chat(u)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
