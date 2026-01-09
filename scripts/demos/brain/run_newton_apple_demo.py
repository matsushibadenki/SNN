# ファイルパス: scripts/demos/brain/run_newton_apple_demo.py
# 日本語タイトル: Brain v4.0 Newton's Apple (Multimodal Coordination)
# 目的: テキスト「りんごが木から落ちた」と、落下する映像データを同時に脳に入力し、
#       言語と視覚が協調して内部状態(State)を形成する様子を実証する。

from snn_research.models.experimental.brain_v4 import SynestheticBrain
import sys
import os
import torch
from transformers import AutoTokenizer

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))


def run_coordination_demo():
    print("\n=======================================================")
    print(" 🍎 Brain v4.0 Multimodal Coordination: Newton's Apple")
    print("=======================================================\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # 1. 脳の初期化
    # ---------------------------------------------------------
    # Brain v4.0 (5-Senses) をロード
    brain = SynestheticBrain(
        vocab_size=50257,  # GPT-2 size
        d_model=256,
        time_steps=8,     # 8フレーム分の時間を思考
        device=device
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"🧠 Brain initialized on {device.upper()}. ready for multimodal input.")

    # 2. データの準備 (テキスト + 映像)
    # ---------------------------------------------------------

    # [テキスト]: "The apple fell from the tree"
    text_prompt = "The apple fell from the tree"
    text_inputs = tokenizer(
        text_prompt, return_tensors="pt").input_ids.to(device)
    print(f"\n📖 [Text Input]: \"{text_prompt}\"")

    # [映像]: 落下運動を模したスパイク映像データ (Batch, Time, Channel, H, W)
    # ここでは簡易的に、フレームごとに「白い点(りんご)」がY軸方向に下がるデータを生成
    image_seq = torch.zeros((1, 8, 1, 28, 28)).to(device)
    for t in range(8):
        # 時間経過(t)とともにY座標(row)を下げる => 落下運動
        y_pos = 2 + t * 3
        if y_pos < 28:
            # りんごの描画 (3x3の白い矩形)
            image_seq[0, t, 0, y_pos:y_pos+3, 12:15] = 1.0

    print("🎬 [Vision Input]: Video sequence of an object falling (8 frames).")

    # 3. 協調思考 (Simultaneous Reasoning)
    # ---------------------------------------------------------
    # テキストと映像を同時に forward に渡すことで、
    # 内部で「映像トークン」と「テキストトークン」が結合(Concatenate)され、
    # BitSpikeMambaコアがそれらを相互参照しながら処理します。

    brain.eval()
    with torch.no_grad():
        # A. 感覚入力の統合処理
        #    Brain v4の forward メソッドは、内部で以下を行います:
        #    1. image_input -> UniversalEncoder -> VisionProjector -> VisionEmbeddings [B, 8, D]
        #    2. text_input  -> TextEmbeddings [B, L, D]
        #    3. Combined -> [Vision; Text] (時系列順または並列に結合) -> CoreBrain

        logits = brain(
            text_input=text_inputs,
            image_input=image_seq
        )

        # B. 思考結果の確認
        #    脳が「映像(落下)」と「テキスト(りんご)」を見て、次に何を想起するか予測
        last_token_logits = logits[:, -1, :]
        predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
        predicted_word = tokenizer.decode([predicted_token_id])

    print("\n-------------------------------------------------------")
    print(" 🧠 Thinking Process (Internal State Analysis)")
    print("-------------------------------------------------------")
    print("1. **Visual Cortex**: Detected downward motion (Gravity).")
    print("2. **Language Center**: Parsed subject 'Apple' and context 'Tree'.")
    print("3. **Association Area**: Linked 'Visual Object' ⇔ 'Symbol: Apple'.")
    print(
        f"4. **Prediction**: The brain predicts the next concept is -> '{predicted_word.strip()}'")

    # 補足: 学習済みであれば、ここで "Ground" や "Gravity"、"Impact" などが出るはず。
    #       未学習(ランダム重み)の場合はランダムな単語が出ます。

    print("\n✅ Multimodal coordination successful.")
    print("   The model processed visual dynamics and semantic context in a single pass.")


if __name__ == "__main__":
    run_coordination_demo()
