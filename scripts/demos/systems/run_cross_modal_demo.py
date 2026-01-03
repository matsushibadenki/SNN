# ファイルパス: scripts/run_cross_modal_demo.py
# Title: Cross-Modal Association Demo [Optimized]
# Description:
#   発火率を確保するためのパラメータチューニング適用済み。

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.networks.liquid_association_cortex import LiquidAssociationCortex
from snn_research.learning_rules.stdp import STDP
from snn_research.io.universal_encoder import UniversalSpikeEncoder

def run_demo():
    print("🧠 --- Phase 9-10: Cross-Modal Association Demo (Hearing Colors) ---")
    
    # 設定
    DEVICE = "cpu"
    TIME_STEPS = 50
    BATCH_SIZE = 1
    
    # パラメータ調整
    LEARNING_RATE = 0.02
    A_PLUS = 0.1
    EPOCHS = 30
    INPUT_SCALE = 20.0 # 入力信号を強力に増幅
    THRESHOLD = 0.5    # 発火閾値を下げる
    
    # 入力次元
    DIM_VISUAL = 100
    DIM_AUDIO = 50
    RESERVOIR_SIZE = 500
    
    # 1. 初期化
    stdp_rule = STDP(
        learning_rate=LEARNING_RATE,
        a_plus=A_PLUS,
        a_minus=0.05,
        tau_trace=20.0
    )
    
    lac = LiquidAssociationCortex(
        num_visual_inputs=DIM_VISUAL,
        num_audio_inputs=DIM_AUDIO,
        num_text_inputs=10,
        num_somato_inputs=10,
        reservoir_size=RESERVOIR_SIZE,
        sparsity=0.2, # 少し密にする
        tau=5.0,      # 時定数を長くして記憶を保ちやすくする
        threshold=THRESHOLD,
        input_scale=INPUT_SCALE,
        learning_rule=stdp_rule
    ).to(DEVICE)
    
    encoder = UniversalSpikeEncoder(time_steps=TIME_STEPS, device=DEVICE)
    
    print(f"✅ System Initialized. Input Scale={INPUT_SCALE}, Threshold={THRESHOLD}")

    # 2. パターン生成
    torch.manual_seed(42)
    vis_A = torch.rand(BATCH_SIZE, DIM_VISUAL) > 0.7
    aud_A = torch.rand(BATCH_SIZE, DIM_AUDIO) > 0.7
    
    vis_A_spikes = encoder.encode(vis_A.float(), 'image', 'rate')
    aud_A_spikes = encoder.encode(aud_A.float(), 'audio', 'rate')
    
    # 3. ベースライン取得
    print("\n🎧 [Pre-Training] Testing Audio-only response...")
    lac.eval()
    lac.reset_state()
    
    pre_activity = []
    for t in range(TIME_STEPS):
        out = lac(visual_spikes=None, audio_spikes=aud_A_spikes[:, t, :])
        pre_activity.append(out.detach())
    pre_activity = torch.stack(pre_activity).squeeze(1)
    
    lac.reset_state()
    target_vis_activity = []
    for t in range(TIME_STEPS):
        # ターゲットは視覚+音声同時入力時のリザーバ状態とする（より自然な連合）
        # または純粋な視覚想起を目指すなら視覚のみでも良いが、ここでは「Aという概念」全体をターゲットとする
        out = lac(visual_spikes=vis_A_spikes[:, t, :], audio_spikes=aud_A_spikes[:, t, :])
        target_vis_activity.append(out.detach())
    target_vis_activity = torch.stack(target_vis_activity).squeeze(1)

    sim_pre = F.cosine_similarity(pre_activity.mean(dim=0).unsqueeze(0), target_vis_activity.mean(dim=0).unsqueeze(0))
    print(f"   Similarity to Target Concept (Before Learning): {sim_pre.item():.4f}")

    # 4. 連合学習フェーズ
    print(f"\n🔗 [Training] Associating Visual-A with Audio-A ({EPOCHS} Epochs)...")
    lac.train()
    
    for epoch in range(EPOCHS):
        lac.reset_state()
        total_spikes = 0
        for t in range(TIME_STEPS):
            out = lac(
                visual_spikes=vis_A_spikes[:, t, :], 
                audio_spikes=aud_A_spikes[:, t, :]
            )
            total_spikes += out.sum().item()
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}: Total Reservoir Spikes: {total_spikes}")

    # 5. 想起テスト
    print("\n🎨 [Post-Training] Testing 'Hearing Colors' (Audio-only Input)...")
    lac.eval()
    lac.reset_state()
    
    post_activity = []
    for t in range(TIME_STEPS):
        # 音声のみ入力
        out = lac(visual_spikes=None, audio_spikes=aud_A_spikes[:, t, :])
        post_activity.append(out.detach())
    post_activity = torch.stack(post_activity).squeeze(1)
    
    # 6. 結果検証
    sim_post = F.cosine_similarity(post_activity.mean(dim=0).unsqueeze(0), target_vis_activity.mean(dim=0).unsqueeze(0))
    
    print(f"   Similarity to Target Concept (After Learning):  {sim_post.item():.4f}")
    
    improvement = sim_post.item() - sim_pre.item()
    print(f"\n📈 Association Improvement: {improvement:+.4f}")
    
    # 判定基準
    if improvement > 0.05:
        print("✅ SUCCESS: Cross-modal association detected.")
    else:
        print("⚠️ WARNING: Association weak.")

    # 可視化
    try:
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(target_vis_activity.T.numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
        plt.title("Target Concept Activity")
        plt.xlabel("Time")
        plt.ylabel("Neurons")
        
        plt.subplot(1, 3, 2)
        plt.imshow(pre_activity.T.numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
        plt.title(f"Audio Response (Pre)\nSim: {sim_pre.item():.2f}")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(post_activity.T.numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
        plt.title(f"Audio Response (Post)\nSim: {sim_post.item():.2f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("workspace/results/cross_modal_demo.png")
        print("   Raster plot saved to results/cross_modal_demo.png")
    except Exception as e:
        print(f"   (Plotting skipped: {e})")

if __name__ == "__main__":
    run_demo()