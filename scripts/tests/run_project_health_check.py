# scripts/tests/run_project_health_check.py
# ディレクトリ: scripts/tests
# 日本語タイトル: プロジェクト健全性チェック
# 説明: 主要モジュールのインポート、ディレクトリ構造、依存関係をチェックする。
#       感情(Emotion)・身体性(Embodiment)に加え、社会性(Social)・生物学(Bio)・進化(Evolution)モジュールも対象。

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
        # コア
        "snn_research.core.snn_core",
        "snn_research.core.neurons.da_lif_node",
        # トランスフォーマーモデル
        "snn_research.models.transformer.spikformer",
        # ハイブリッドモデル
        "snn_research.models.hybrid.concept_spikformer",
        "snn_research.models.hybrid.emotional_concept_brain", # [追加] 感情・概念脳
        # 身体性エージェント
        "snn_research.models.embodied.emotional_agent", # [追加] 身体性
        # 生物学的モデル
        "snn_research.models.bio.visual_cortex", # [追加] 視覚野
        # 認知アーキテクチャ
        "snn_research.cognitive_architecture.neuro_symbolic_bridge",
        "snn_research.cognitive_architecture.amygdala", # [追加] 扁桃体
        "snn_research.cognitive_architecture.hippocampus", # [追加] 海馬
        "snn_research.cognitive_architecture.prefrontal_cortex", # [追加] 前頭前野
        # 社会性・コミュニケーション
        "snn_research.social.theory_of_mind", # [追加] 心の理論
        "snn_research.social.synesthetic_dialogue", # [追加] 共感覚対話
        # 進化・適応
        "snn_research.evolution.structural_plasticity", # [追加] 構造的可塑性
        # 蒸留
        "snn_research.distillation.knowledge_distillation_manager", # [追加] 蒸留
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
        "snn_research/models/bio",      # [追加] 生物学的モデル
        "snn_research/cognitive_architecture",
        "snn_research/social",          # [追加] 社会性
        "snn_research/systems",         # [追加] システム統合
        "snn_research/evolution",       # [追加] 進化
        "snn_research/distillation",    # [追加] 蒸留
        "snn_research/training",
        "scripts/experiments",
        "tests/models",
        "tests/cognitive_architecture", # [追加]
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