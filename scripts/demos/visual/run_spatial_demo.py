# ファイルパス: scripts/run_spatial_demo.py
# Title: 空間認識・マルチモーダルデモ (修正版)
# Description:
#   人工脳が画像から物体を検出し、その位置情報（空間コンテキスト）を認識するデモ。
#   VisualCortexの検出結果を確実に取得するため、デモ用に再検出を行うロジックを追加。

import sys
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

def generate_dummy_image(filename: str = "spatial_test_image.png"):
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    return filename

def main():
    print("📍 空間認識 & Context-Aware Routing デモを開始します...\n")

    # 1. コンテナ初期化
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")
    
    brain = container.artificial_brain()
    encoder = container.spike_encoder() # SpikeEncoderインスタンス

    # 2. 画像入力と空間認識
    print(">>> Step 1: 画像入力と物体検出")
    image_path = generate_dummy_image()
    brain.run_cognitive_cycle(image_path)
    
    # --- 修正: Workspaceから消えている可能性があるため、デモ用に再取得 ---
    # 画像をロード
    image_tensor = brain.image_transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    
    # VisualCortexのメソッドを直接呼んで検出結果を取得 (デモ用)
    objects = brain.visual_cortex.detect_objects(image_tensor)
    
    if objects:
        print(f"\n📊 検出されたオブジェクト数: {len(objects)}")
        
        # 3. 空間情報のスパイクエンコーディング (Spatial Coding)
        print("\n>>> Step 2: 空間情報のスパイク化 (SpikeEncoder)")
        for obj in objects:
            label = obj["label"]
            bbox = obj["bbox"] # [x, y, w, h]
            
            # SpikeEncoderを使って空間情報をエンコード
            # sensory_receptorを通さず直接エンコーダをテスト
            spatial_spikes = encoder.encode(
                {"type": "spatial", "content": bbox}, 
                duration=16
            )
            
            print(f"  - Object: {label}")
            print(f"    BBox: {bbox}")
            print(f"    Spatial Spikes Shape: {spatial_spikes.shape}")
            print(f"    Active Neurons: {spatial_spikes.sum().item()} / {spatial_spikes.numel()}")
            
    else:
        print("⚠️ 物体が検出されませんでした。")

    # 4. Context-Aware Routing のシミュレーション
    print("\n>>> Step 3: Context-Aware Routing Simulation")
    from snn_research.models.experimental.moe_model import ContextAwareSpikingRouter
    
    input_dim = 128
    num_experts = 3
    
    # "type": "lif" を削除し、有効なパラメータのみ渡す
    router = ContextAwareSpikingRouter(input_dim, num_experts, {})
    
    # ダミー入力
    text_input = torch.randn(1, input_dim) # テキスト
    
    # ケースA: 視覚コンテキストなし
    weights_no_ctx = router(text_input, None)
    print(f"  Routing Weights (No Context): {weights_no_ctx.detach().numpy()}")
    
    # ケースB: 視覚コンテキストあり
    # デモ用にダミーコンテキストを作成
    visual_ctx = torch.randn(1, input_dim)
            
    weights_with_ctx = router(text_input, visual_ctx)
    print(f"  Routing Weights (With Visual Context): {weights_with_ctx.detach().numpy()}")
    
    # 差分を確認
    diff = (weights_with_ctx - weights_no_ctx).abs().sum().item()
    print(f"  -> Context Influence (Difference): {diff:.4f}")
    if diff > 0:
        print("  ✅ 視覚コンテキストがルーティングに影響を与えました。")

    # 終了処理
    if os.path.exists(image_path):
        os.remove(image_path)
        
    print("\n🎉 デモ完了")

if __name__ == "__main__":
    main()