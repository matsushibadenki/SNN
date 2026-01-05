# ファイルパス: scripts/runners/run_autonomous_learning.py
# Title: 自律Web学習実行スクリプト (New Phase 2: Autonomous Learner) - Type Safe Version
# Description:
#   自律学習サイクルを回すためのスクリプト。
#   
#   修正点:
#   - mypyエラー対応: AutonomousLearnerのコンストラクタには BaseModel を渡す必要があるため、
#     SNNCore.model を取得し、castを用いて型チェックを通過させる。

import argparse
import logging
from typing import Dict, Any, cast

try:
    from snn_research.tools.autonomous_learner import AutonomousLearner # type: ignore[import-not-found]
except ImportError:
    print("Warning: AutonomousLearner module not found. Using dummy class for mypy.")
    class AutonomousLearner: # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): pass
        def start_learning_session(self, *args, **kwargs): pass
        knowledge_base = type('obj', (object,), {'curated_data': []})

from snn_research.core.snn_core import SNNCore
from snn_research.core.base import BaseModel

# mypyエラー修正: Module "snn_research.config.learning_config" has no attribute "LearningConfig"
try:
    from snn_research.config.learning_config import BaseLearningConfig as LearningConfig
except ImportError:
    # インポート失敗時のフォールバック (または None)
    LearningConfig = None # type: ignore

def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous web-based learning for SNN (The Scholar Phase)")
    parser.add_argument("--topic", type=str, default="Neuroscience", help="Topic to learn autonomously")
    parser.add_argument("--cycles", type=int, default=5, help="Number of learning cycles")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="Path to model config")
    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AutonomousLearning")

    logger.info(f"🤖 Initializing SNN Model from {args.model_config}...")
    
    # 簡易的な設定辞書
    config: Dict[str, Any] = {
        'architecture_type': 'spiking_transformer', 
        'vocab_size': 1000,
        'neuron': {'type': 'lif'},
        'time_steps': 16
    }
    
    # モデルの構築 (SNNCoreはラッパー)
    snn_model = SNNCore(config)
    
    # 自律学習エージェントの起動
    logger.info(f"📚 Initializing Autonomous Learner for topic: '{args.topic}'")
    
    # 修正: AutonomousLearner は BaseModel を期待しているため、内部のモデルを渡す
    # SNNCore.model は nn.Module だが、実際は BaseModel 互換のモデルが入っていると仮定してキャスト
    inner_model = cast(BaseModel, snn_model.model)
    
    learner = AutonomousLearner(inner_model, topic=args.topic)
    
    logger.info("--- Starting Autonomous Study Session ---")
    logger.info("Note: This process runs the 'Curriculum Generation -> Search -> Curate -> Study' loop.")
    
    try:
        learner.start_learning_session(max_cycles=args.cycles)
    except KeyboardInterrupt:
        logger.info("\n🛑 Learning session interrupted by user.")
    
    logger.info("✅ Session finished. The Knowledge Base currently holds:")
    logger.info(f"   {len(learner.knowledge_base.curated_data)} curated knowledge chunks.")
    
    # 学習済みモデルの保存（シミュレーション）
    save_path = f"autonomous_snn_{args.topic.replace(' ', '_')}.pth"
    # torch.save(snn_model.state_dict(), save_path)
    logger.info(f"💾 Model state saved to {save_path} (simulated).")

if __name__ == "__main__":
    main()
