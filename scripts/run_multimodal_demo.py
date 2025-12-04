# ファイルパス: scripts/run_multimodal_demo.py
# Title: マルチモーダル人工脳デモ
# Description:
#   画像入力とテキスト入力を交互に人工脳に与え、
#   VisualCortexと通常の知覚野が適切に切り替わって処理される様子を確認する。
#   Cross-Modal Injection（視覚情報の言語野への注入）の動作もシミュレートする。

import sys
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

def generate_dummy_image(filename: str = "test_image.png"):
    """デモ用のダミー画像を生成する"""
    # ランダムなノイズ画像を作成
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"🖼️ ダミー画像を生成しました: {filename}")
    return filename

def main():
    print("🧠 マルチモーダル人工脳デモを開始します...")

    # 1. DIコンテナの初期化
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")
    
    brain = container.artificial_brain()
    
    # 2. テキスト入力の処理 (通常のフロー)
    print("\n>>> Test 1: Text Input Processing")
    text_input = "これは言語処理のテストです。"
    brain.run_cognitive_cycle(text_input)
    
    # 内部状態の確認 (テキスト知覚)
    perception_info = brain.workspace.get_information("perception")
    if perception_info:
        print("  ✅ 知覚野 (SOM) がテキスト特徴を処理しました。")
    else:
        print("  ⚠️ 知覚野からの情報がありません。")

    # 3. 画像入力の処理 (Visual Cortexフロー)
    print("\n>>> Test 2: Image Input Processing")
    image_path = generate_dummy_image()
    
    # 画像パスを渡す (SensoryReceptorが画像をロード)
    brain.run_cognitive_cycle(image_path)
    
    # 内部状態の確認 (視覚知覚)
    visual_info = brain.workspace.get_information("visual_cortex")
    if visual_info:
        print("  ✅ 視覚野 (Visual Cortex) が画像を処理しました。")
        print(f"     - 生成されたコンテキスト埋め込み形状: {visual_info.get('context_embeds').shape}")
        print("     (この埋め込みベクトルが言語野へ注入され、視覚に基づいた思考が可能になります)")
    else:
        print("  ⚠️ 視覚野からの情報がありません。")

    # 4. 知識の訂正 (ユーザーインタラクション)
    print("\n>>> Test 3: Knowledge Correction")
    brain.correct_knowledge("ArtificialBrain", "Multimodal System", reason="Upgrade to Phase 3")
    
    # 終了処理
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"\n🗑️ 一時ファイルを削除しました: {image_path}")
        
    print("\n🎉 デモ完了")

if __name__ == "__main__":
    main()