# scripts/tests/run_project_health_check.py
# ディレクトリ: scripts/tests
# 日本語タイトル: プロジェクト健全性チェック
# 説明: 主要モジュールのインポート、ディレクトリ構造、依存関係をチェックする。
#       感情(Emotion)・身体性(Embodiment)モジュールを追加。

import sys
import os
import importlib
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("HealthCheck")

def check_imports():
    """主要モジュールがインポート可能かチェック"""
    modules_to_check = [
        "snn_research.core.snn_core",
        "snn_research.core.neurons.da_lif_node",
        "snn_research.models.transformer.spikformer",
        "snn_research.models.hybrid.concept_spikformer",
        # [追加] 感情・概念脳
        "snn_research.models.hybrid.emotional_concept_brain",
        # [追加] 身体性エージェント
        "snn_research.models.embodied.emotional_agent",
        # [追加] 認知アーキテクチャ
        "snn_research.cognitive_architecture.neuro_symbolic_bridge",
        "snn_research.cognitive_architecture.amygdala",
        # トレーナー
        "snn_research.training.trainers.concept_augmented_trainer",
    ]
    
    all_passed = True
    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ Import successful: {module_name}")
        except ImportError as e:
            logger.error(f"❌ Import failed: {module_name} -> {e}")
            all_passed = False
        except Exception as e:
            logger.error(f"❌ Error importing {module_name} -> {e}")
            all_passed = False
            
    return all_passed

def check_directories():
    """必要なディレクトリ構造が存在するかチェック"""
    required_dirs = [
        "snn_research/core",
        "snn_research/models/transformer",
        "snn_research/models/hybrid",
        "snn_research/models/embodied", # [追加]
        "snn_research/cognitive_architecture",
        "snn_research/training",
        "scripts/experiments",
        "tests/models",
    ]
    
    root_path = os.path.join(os.path.dirname(__file__), "../../")
    all_passed = True
    
    for d in required_dirs:
        full_path = os.path.join(root_path, d)
        if os.path.isdir(full_path):
            logger.info(f"✅ Directory exists: {d}")
        else:
            logger.error(f"❌ Directory missing: {d}")
            all_passed = False
            
    return all_passed

def main():
    logger.info("Starting Project Health Check...")
    
    imports_ok = check_imports()
    dirs_ok = check_directories()
    
    if imports_ok and dirs_ok:
        logger.info("\n🎉 All health checks passed! System is ready.")
        sys.exit(0)
    else:
        logger.error("\n⚠️ Some health checks failed. Please review the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()