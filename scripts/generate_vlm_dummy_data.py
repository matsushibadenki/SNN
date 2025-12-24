# ファイルパス: scripts/generate_vlm_dummy_data.py
# Title: VLM学習用ダミーデータ生成
# Description:
#   ランダムなノイズ画像とテキストキャプションのペアを作成し、
#   train_spiking_vlm.py で読み込める JSONL 形式で保存します。

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_dummy_data(output_dir="data/vlm_dummy", num_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "train_data.jsonl")
    
    print(f"🎨 Generating {num_samples} dummy samples in {output_dir}...")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_samples)):
            # 1. 画像生成 (ランダムノイズ)
            img_filename = f"image_{i:04d}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            
            # ランダムな色画像
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(img_path)
            
            # 2. テキスト生成
            captions = [
                "A spiking neural network processing visual data.",
                "An artificial brain dreaming of electric sheep.",
                "Neuromorphic hardware accelerating AI inference.",
                "A robot looking at a futuristic city.",
                "Digital synapses firing in a complex pattern."
            ]
            text = random.choice(captions)
            
            # 3. JSONL書き込み
            # ImageTextDatasetは {'image': path, 'text': text} を期待
            # imageパスはJSONLからの相対パスまたは絶対パス
            entry = {
                "image": img_filename,
                "text": text,
                "label": None # 生成タスクなのでラベルはなし（またはNext Token用）
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Data generation complete: {jsonl_path}")

if __name__ == "__main__":
    generate_dummy_data()