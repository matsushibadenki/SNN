# scripts/experiments/brain/run_emotional_value_learning.py
# 目的: 「感情＝価値関数」というイリヤ・サツケバーの仮説を検証する。
# 外部報酬が希薄な環境で、Amygdalaの「直感」が学習を加速させるかを確認する。

import torch
import sys
import os

# パス設定など...
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.learning_rules.reward_modulated_stdp import EmotionModulatedSTDP
# 既存のSNNコアなどをインポート

def run_experiment():
    print(">>> Ilya's Hypothesis Experiment: Emotion as Value Function <<<")
    
    # モデルの構築
    # 入力: 視覚情報など
    amygdala = Amygdala(input_dim=128)
    learning_rule = EmotionModulatedSTDP(learning_rate=0.01)
    
    # シミュレーションループ
    for episode in range(100):
        # ... 環境からの入力 ...
        sensory_input = torch.randn(1, 128) # ダミー入力
        
        # 1. 扁桃体による直感的な価値判断
        # 「これ、なんか良さそう」「これ、なんかヤバそう」を即座に判定
        with torch.no_grad():
            internal_value = amygdala(sensory_input).item()
        
        # 2. 行動選択 (Prefrontal Cortex等が担当する想定)
        # ここでは簡易的にランダム + 価値によるバイアス
        action = "explore" if internal_value > 0 else "avoid"
        
        # 3. 環境からのフィードバック (遅延報酬)
        # ほとんどの場合 0 (報酬なし)。たまに 1 (成功)
        external_reward = 0.0 
        if episode % 20 == 0: # 稀に報酬がある
            external_reward = 1.0
            print(f"  [Episode {episode}] External Reward Received!")
            
            # 4. 価値関数の更新
            # 「さっきの直感は正しかった」と学習させる
            amygdala.update_value_function(sensory_input, external_reward)
        
        # 5. SNNの学習 (STDP)
        # 外部報酬がない時でも、internal_value (感情) があればシナプスが更新される！
        # これが「報酬なし学習」「一般化」の鍵。
        # weights = learning_rule.update(..., external_reward, internal_value)
        
        print(f"  [Episode {episode}] Value(Emotion): {internal_value:.4f}, Action: {action}")

if __name__ == "__main__":
    run_experiment()