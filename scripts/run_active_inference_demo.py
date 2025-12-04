# ファイルパス: scripts/run_active_inference_demo.py
# Title: Active Inference & Embodiment Demo (Syntax Fixed)
# Description:
#   ROADMAP Phase 4 の主要機能である「能動的推論エージェント」の動作を検証するスクリプト。
#   - 修正: 末尾の不要な '}' を削除。

import sys
import os
from pathlib import Path
import torch
import asyncio
import logging

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ActiveInferenceDemo")

def main():
    logger.info("🤖 Deep Active Inference Demo を開始します...")
    
    # 1. DIコンテナの初期化
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    # ActiveInferenceAgent用のモデル設定を含むコンフィグを使用
    container.config.from_yaml("configs/models/small.yaml")
    
    # 2. エージェントの取得
    agent = container.digital_life_form().active_inference_agent
    logger.info("✅ ActiveInferenceAgent を DigitalLifeForm から取得しました。")
    
    # 3. 選好（ゴール）の設定
    # 観測次元数に合わせたターゲット分布を作成
    obs_dim = agent.observation_dim
    target_obs = torch.zeros(obs_dim)
    target_obs[0] = 5.0 # 状態0を強く好む
    agent.set_preference(target_obs)
    
    # 4. 倫理的制約の適用
    # 状態 1 と 3 は「危険」または「非倫理的」として回避させる
    agent.set_ethical_preference(avoid_indices=[1, 3], penalty_strength=10.0)
    
    # 5. インタラクションループ (推論 -> 行動 -> 学習)
    logger.info("\n--- Interaction Loop Start ---")
    
    for step in range(5):
        logger.info(f"\n[Step {step+1}]")
        
        # (A) 観測 (シミュレーション)
        # 環境からの入力を模倣 (ランダムな観測)
        observation = torch.randn(1, obs_dim)
        logger.info(f"  👁️ Observation received (shape: {observation.shape})")
        
        # (B) 状態推論 (Perception)
        # 観測から現在の信念(Belief)を更新
        belief = agent.infer_state(observation)
        # logger.info(f"  🧠 State inferred (Belief mean: {belief.mean().item():.4f})")
        
        # (C) 行動選択 (Action Selection)
        # 期待自由エネルギー G = Risk + Ambiguity を最小化する行動を選ぶ
        action_idx = agent.select_action()
        # (内部で G値 のログが出力されるはず)
        
        # (D) モデル更新 (Learning)
        # 観測結果に基づいて生成モデルを微調整
        agent.update_model(observation, action=action_idx)
        
    logger.info("\n✅ Demo Complete. Active Inference loop verified.")

if __name__ == "__main__":
    main()
