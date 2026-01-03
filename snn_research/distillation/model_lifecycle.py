# ファイルパス: snn_research/distillation/model_lifecycle.py
# Title: Model Lifecycle Manager (相対パス対応版)
# Description:
# - モデルレジストリ内のモデルを分析し、ライフサイクル（アーカイブ、削除、MoE化）を管理する。
# - FrankenMoEの設定生成時に、絶対パスではなくプロジェクトルートからの相対パスを使用するように改善。

import os
import shutil
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelLifecycleManager:
    """
    モデルのライフサイクル（選抜、整理、統合）を管理するクラス。
    """
    def __init__(self, registry: ModelRegistry, archive_dir: str = "workspace/runs/archived_models"):
        self.registry = registry
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        # プロジェクトルートの取得 (このファイルの2つ上のディレクトリと仮定)
        self.project_root = Path(__file__).resolve().parent.parent.parent

    async def cleanup_models(self, keep_top_k: int = 3, metric: str = "accuracy"):
        """
        各タスクについて、性能の低い古いモデルをアーカイブ（または削除）する。
        """
        logger.info(f"🧹 Cleaning up models (keeping top {keep_top_k} per task based on {metric})...")
        
        all_models = await self.registry.list_models()
        
        # タスクごとにモデルをグループ化
        tasks: Dict[str, List[Dict[str, Any]]] = {}
        for m in all_models:
            task = m.get('task_description', 'unknown')
            if task not in tasks:
                tasks[task] = []
            tasks[task].append(m)

        for task, models in tasks.items():
            # 性能でソート (降順)
            models.sort(key=lambda x: x.get('metrics', {}).get(metric, 0.0), reverse=True)
            
            # 上位K個以外をアーカイブ対象とする
            to_archive = models[keep_top_k:]
            
            for m in to_archive:
                model_path_str = m.get('model_path') or m.get('path')
                if model_path_str and os.path.exists(model_path_str):
                    src_path = Path(model_path_str)
                    filename = src_path.name
                    dest_path = self.archive_dir / f"{task}_{filename}"
                    
                    try:
                        # ファイルを移動
                        shutil.move(src_path, dest_path)
                        logger.info(f"  - Archived: {src_path} -> {dest_path}")
                        # レジストリ情報の更新が必要だが、現在のSimpleRegistryは上書きが難しいため
                        # ここではログ出力にとどめる。（実運用ではレジストリDBの更新が必要）
                    except Exception as e:
                        logger.error(f"Failed to archive {src_path}: {e}")

    async def create_franken_moe_config(self, task_keywords: List[str], output_config_path: str) -> Optional[Dict[str, Any]]:
        """
        指定されたキーワードに関連するタスクのベストモデルを集め、
        FrankenMoE用の設定ファイルを生成する。
        """
        logger.info(f"🧟 Creating FrankenMoE config for keywords: {task_keywords}")
        
        experts = []
        
        for keyword in task_keywords:
            # キーワードにマッチするタスクのベストモデルを検索
            found = await self.registry.find_models_for_task(keyword, top_k=1)
            if found:
                best_model = found[0]
                experts.append(best_model)
                logger.info(f"  - Added expert for '{keyword}': {best_model.get('model_id')} (Acc: {best_model.get('metrics', {}).get('accuracy', 0.0):.4f})")
            else:
                logger.warning(f"  - No expert found for keyword '{keyword}'.")
                
        if not experts:
            logger.error("No experts found. Cannot create MoE config.")
            return None
            
        # MoE設定の構築
        # ベースとなるハイパーパラメータは最初のエキスパートから借用
        base_config = experts[0].get('config', {})
        
        # パスを相対パスに変換するヘルパー
        def to_relative_path(path_str: Optional[str]) -> str:
            if not path_str:
                return "None"
            try:
                abs_path = Path(path_str).resolve()
                return str(abs_path.relative_to(self.project_root))
            except ValueError:
                # プロジェクトルート外の場合はそのまま返す
                return path_str

        moe_config = {
            "architecture_type": "franken_moe",
            "d_model": base_config.get("d_model", 128),
            "expert_configs": [e.get('config') for e in experts],
            # ここで相対パス変換を適用
            "expert_checkpoints": [to_relative_path(e.get('model_path') or e.get('path')) for e in experts],
            # ニューロン設定などは共通化
            "neuron": base_config.get("neuron", {"type": "lif"}),
            "time_steps": base_config.get("time_steps", 16)
        }
        
        # YAML/JSONとして保存
        import yaml
        with open(output_config_path, 'w') as f:
            yaml.dump({"model": moe_config}, f)
            
        logger.info(f"✅ FrankenMoE config saved to: {output_config_path}")
        return moe_config