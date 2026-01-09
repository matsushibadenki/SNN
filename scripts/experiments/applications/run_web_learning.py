# ファイルパス: scripts/experiments/applications/run_web_learning.py
# タイトル: 自律型 Web学習ランナー (Autonomous Web Learning)
# 目的: Webクローラーを利用して自律的にトピックを探索し、継続的に知識蒸留学習を行う環境を提供する。
# 内容:
#   - 自律学習ループ（Curiosity Loop）の実装
#   - 学習した内容から次の興味（トピック）を生成する機能
#   - 継続的なモデルのアップデートと評価

# ◾️ DistillationTrainer をインポート
from snn_research.training.trainers import DistillationTrainer
from app.containers import TrainingContainer  # DIコンテナを利用
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from app.services.web_crawler import WebCrawler
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
import torch
from typing import Optional, Any, Dict, List
import asyncio
import argparse
import sys
import os
import random

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加
# ------------------------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------


def extract_next_topics(data_path: str, current_topic: str) -> List[str]:
    """
    収集したデータから、次に学習すべきトピック（キーワード）を抽出する。
    簡易的な「好奇心」の実装。
    """
    if not os.path.exists(data_path):
        return []

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # 簡易的なキーワード抽出（本来はNLPモデルを使うべき箇所）
    # ここでは、AI/脳科学に関連しそうな重要単語をリストアップし、出現有無で判断
    potential_keywords = [
        "synapse", "plasticity", "energy", "quantum", "consciousness",
        "robotics", "evolution", "chaos", "entropy", "optimization",
        "transformer", "neuromorphic", "spiking", "dopamine"
    ]

    found_topics = []
    for kw in potential_keywords:
        if kw in text and kw != current_topic.lower():
            found_topics.append(kw)

    # 見つからなければランダムな未知語（デモ用）
    if not found_topics:
        found_topics = ["unknown_phenomenon", "future_tech", "deep_brain"]

    return list(set(found_topics))


def setup_distillation_manager(container: TrainingContainer) -> KnowledgeDistillationManager:
    """DIコンテナから学習マネージャを構築して返すヘルパー関数"""
    device: str = container.device()
    student_model: torch.nn.Module = container.snn_model()
    optimizer: torch.optim.Optimizer = container.optimizer(
        params=student_model.parameters())

    # 設定ファイルに基づき、スケジューラを条件付きで作成
    scheduler: Optional[LRScheduler] = container.scheduler(
        optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

    # Trainerの構築
    distillation_trainer: "DistillationTrainer" = container.distillation_trainer(
        model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1
    )

    # Managerの構築
    manager_config: DictConfig = container.config()
    distillation_manager = KnowledgeDistillationManager(
        student_model=student_model,
        # type: ignore[arg-type]
        trainer=distillation_trainer,
        teacher_model_name=container.config.training.gradient_based.distillation.teacher_model(),
        tokenizer_name=container.config.data.tokenizer_name(),
        model_registry=container.model_registry(),
        device=device,
        config=manager_config
    )
    return distillation_manager


async def run_autonomous_loop(initial_topic: str, start_url: str, max_cycles: int):
    """
    自律的な学習ループを実行する。
    Crawl -> Learn -> Generate Next Topic -> Repeat
    """
    print("\n" + "="*50)
    print(f" 🚀 Starting Autonomous Learning Loop (Max Cycles: {max_cycles})")
    print("="*50)

    current_topic = initial_topic
    current_url = start_url
    known_topics = set([initial_topic])

    # DIコンテナの初期化（ループ外でモデルを保持する場合はここで行う）
    container = TrainingContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml("configs/models/medium.yaml")

    # 学習マネージャのセットアップ
    distillation_manager = setup_distillation_manager(container)
    student_config_dict: Dict[str, Any] = container.config.model.to_dict()
    crawler = WebCrawler()

    for cycle in range(1, max_cycles + 1):
        print(f"\n🌀 [Cycle {cycle}/{max_cycles}] Topic: '{current_topic}'")

        # --- 1. Crawl ---
        crawled_data_path = crawler.crawl(
            start_url=current_url,
            max_pages=3,
            topic_filter=current_topic
        )

        if not os.path.exists(crawled_data_path) or os.path.getsize(crawled_data_path) == 0:
            print("❌ Crawling failed or empty. Skipping cycle.")
            continue

        # --- 2. Learn ---
        print(f"    🧠 Learning from gathered data about '{current_topic}'...")
        await distillation_manager.run_on_demand_pipeline(
            task_description=current_topic,
            unlabeled_data_path=crawled_data_path,
            force_retrain=True,
            student_config=student_config_dict
        )

        # --- 3. Plan Next ---
        # 収集したデータから次の興味（トピック）を見つける
        next_candidates = extract_next_topics(crawled_data_path, current_topic)

        # 既知のトピックを除外
        new_candidates = [t for t in next_candidates if t not in known_topics]

        if new_candidates:
            next_topic = random.choice(new_candidates)
            print(
                f"    💡 Curiosity triggered! Found interesting new topic: '{next_topic}'")
        else:
            # 新しいものが見つからなければ、少し視点を変える
            next_topic = "general_intelligence"
            print(
                f"    🤔 No new topics found. Returning to base concept: '{next_topic}'")

        known_topics.add(next_topic)
        current_topic = next_topic

        # URLも動的に変えたいが、デモではWikipedia等の検索URLを模倣するか、モックURLを使用
        current_url = f"https://en.wikipedia.org/wiki/{current_topic.replace(' ', '_')}"

        # サイクル間の休憩
        print("    💤 Sleeping for consolidation (simulated)...")
        await asyncio.sleep(2)

    print("\n" + "="*50)
    print(" 🎉 Autonomous Learning Session Completed.")
    print(f" 📚 Learned Topics: {known_topics}")
    print("="*50)


def main() -> None:
    """
    Webクローラーとオンデマンド学習パイプラインを連携させ、
    全自動で学習環境を回すスクリプト。
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Web Learning Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Neuromorphic Computing",
        help="最初の学習トピック。"
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://en.wikipedia.org/wiki/Neuromorphic_engineering",
        help="開始URL。"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="自律学習ループモードを有効にする（トピックを自動遷移）。"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="自律モード時の最大サイクル数。"
    )

    args: argparse.Namespace = parser.parse_args()

    if args.autonomous:
        # 自律ループモード実行
        asyncio.run(run_autonomous_loop(
            args.topic, args.start_url, args.cycles))
    else:
        # 単発実行モード（従来の挙動）
        print("\n" + "="*20 + " 🌐 Step 1: Web Crawling (Single Shot) " + "="*20)
        crawler = WebCrawler()
        crawled_data_path: str = crawler.crawl(
            start_url=args.start_url, max_pages=5, topic_filter=args.topic)

        if not os.path.exists(crawled_data_path) or os.path.getsize(crawled_data_path) == 0:
            print("❌ データが収集できなかったため、学習を中止します。")
            return

        print("\n" + "="*20 + " 🧠 Step 2: On-demand Learning " + "="*20)
        container = TrainingContainer()
        container.config.from_yaml("configs/templates/base_config.yaml")
        container.config.from_yaml("configs/models/medium.yaml")

        manager = setup_distillation_manager(container)
        student_config_dict: Dict[str, Any] = container.config.model.to_dict()

        asyncio.run(manager.run_on_demand_pipeline(
            task_description=args.topic,
            unlabeled_data_path=crawled_data_path,
            force_retrain=True,
            student_config=student_config_dict
        ))
        print(f"\n🎉 学習完了: トピック「{args.topic}」")


if __name__ == "__main__":
    main()
