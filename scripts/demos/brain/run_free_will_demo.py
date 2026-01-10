# ファイルパス: scripts/demos/brain/run_free_will_demo.py
# 日本語タイトル: Free Will Demo (The Cake Dilemma)
# 目的・内容:
#   「ケーキを食べたい」という強い衝動（Impulse）と、
#   「健康でいたい」という長期的目標（Goal）の葛藤をシミュレーション。
#   AgencyEngineが衝動を抑制（Veto）することで、自由意志の発揮を確認する。

from snn_research.cognitive_architecture.agency_engine import AgencyEngine
import os
import sys
import torch
import logging

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)


def run_free_will_demo():
    print("""
    ============================================================
       🕊️ FREE WILL DEMO (Impulse vs Intention) 🕊️
    ============================================================
    """)

    device = "cpu"
    action_dim = 4
    goal_dim = 32

    # 1. Initialize Engine
    # 学習済みでないため、手動で重みを調整して「特定の条件でVetoする」性格を作る
    agent = AgencyEngine(action_dim=action_dim, hidden_dim=goal_dim).to(device)

    # 手動調整: Goalの特定ビットが立っている時、強いActionを抑制するようにバイアスをかける
    # (デモ用ハック: 本来は学習によって獲得される倫理観)
    with torch.no_grad():
        # Goalの影響力を強める
        agent.evaluator[0].weight.data[:, action_dim:] *= 2.0
        # 全体的にVetoしやすくする
        agent.evaluator[2].bias.data += 0.5

    logger.info("🧠 Brain initialized. Scenario: 'The Cake Dilemma'")

    # 2. Simulation
    # Scenario: 目の前にケーキがある。

    # Case A: ダイエット中 (Goal: Health Priority)
    logger.info("\n🍰 Case 1: You are on a strict diet. (Goal: Health)")
    goal_health = torch.ones(1, goal_dim).to(device)  # Strong health focus

    for i in range(3):
        # 強い衝動 (Eat!)
        impulse = torch.randn(1, action_dim).to(device) * 2.0

        result = agent(impulse, goal_health)

        logger.info(
            f"   Impulse: {result['impulse_strength']:.2f} (Eat!) | Veto Prob: {result['veto_prob']:.2f} -> {result['status']}")

        if result['status'] == "VETOED":
            logger.info(
                "   ✅ SELF-CONTROL: You successfully resisted the cake.")
        else:
            logger.info("   ❌ FAILED: You ate the cake...")

    # Case B: チートデイ (Goal: Pleasure Priority)
    logger.info("\n🎉 Case 2: It's Cheat Day! (Goal: Enjoy)")
    goal_enjoy = torch.zeros(1, goal_dim).to(
        device) - 1.0  # Negative weights to suppress veto

    for i in range(3):
        # 同じ強い衝動
        impulse = torch.randn(1, action_dim).to(device) * 2.0

        result = agent(impulse, goal_enjoy)

        logger.info(
            f"   Impulse: {result['impulse_strength']:.2f} (Eat!) | Veto Prob: {result['veto_prob']:.2f} -> {result['status']}")

        if result['status'] == "EXECUTED":
            logger.info("   😋 YUMMY: You enjoyed the cake guilt-free.")
        else:
            logger.info("   🤔 HMM: You hesitated?")

    logger.info("\n🎉 Free Will Demo Completed.")


if __name__ == "__main__":
    run_free_will_demo()
