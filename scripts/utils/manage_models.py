# ファイルパス: scripts/manage_models.py
# (修正 v2: パス修正 & 補完ロジック追加)
# Title: Model Management CLI
# Description:
# - モデルの整理（クリーンアップ）と、FrankenMoEの構築を行うCLIツール。

import argparse
import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import AppContainer
from snn_research.distillation.model_lifecycle import ModelLifecycleManager

async def cleanup(args):
    container = AppContainer()
    registry = container.model_registry()
    manager = ModelLifecycleManager(registry)
    await manager.cleanup_models(keep_top_k=args.keep)

async def build_moe(args):
    container = AppContainer()
    registry = container.model_registry()
    manager = ModelLifecycleManager(registry)
    
    keywords = args.keywords.split(",")
    
    # --- ▼ 修正: デフォルトパスを configs/models/ に変更 ▼ ---
    output_path = Path(args.output)
    if not output_path.parent.name == "models": # 簡易チェック
        if not output_path.is_absolute() and len(output_path.parts) == 1:
             # ファイル名のみの場合は configs/models/ に配置
             output_path = Path("configs/models") / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # --- ▲ 修正 ▲ ---

    # 構成を作成
    moe_config = await manager.create_franken_moe_config(keywords, str(output_path))
    
    if moe_config:
        # --- ▼ 追加: ルートレベルに必要なパラメータを補完する ▼ ---
        # PlannerSNNなどが参照するパラメータを、最初のエキスパートからコピーしてルートに追加
        if 'expert_configs' in moe_config and len(moe_config['expert_configs']) > 0:
            base_expert = moe_config['expert_configs'][0]
            
            # 補完するキーのリスト
            keys_to_copy = ['num_layers', 'd_state', 'n_head']
            for k in keys_to_copy:
                if k not in moe_config and k in base_expert:
                    moe_config[k] = base_expert[k]
        
        # 上書き保存
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump({"model": moe_config}, f)
        # --- ▲ 追加 ▲ ---
        
        print(f"✅ FrankenMoE config updated with compatibility params at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SNN Model Lifecycle Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # cleanup コマンド
    parser_clean = subparsers.add_parser("cleanup", help="Archive old/low-performance models")
    parser_clean.add_argument("--keep", type=int, default=3, help="Number of best models to keep per task")
    
    # build-moe コマンド
    parser_moe = subparsers.add_parser("build-moe", help="Create FrankenMoE config from existing experts")
    parser_moe.add_argument("--keywords", type=str, required=True, help="Comma-separated keywords to select experts (e.g., 'math,chat,code')")
    parser_moe.add_argument("--output", type=str, default="franken_moe.yaml", help="Output filename (saved in configs/models/)")
    
    args = parser.parse_args()
    
    if args.command == "cleanup":
        asyncio.run(cleanup(args))
    elif args.command == "build-moe":
        asyncio.run(build_moe(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
