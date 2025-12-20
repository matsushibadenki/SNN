# ファイルパス: scripts/register_demo_experts.py
# (新規作成)
# Title: デモ用エキスパート登録スクリプト
# Description: FrankenMoEの動作テストのために、既存のモデルチェックポイントを
#              「科学」や「歴史」の専門家としてモデルレジストリに登録する。

import asyncio
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.distillation.model_registry import SimpleModelRegistry

async def main():
    # レジストリの初期化
    registry = SimpleModelRegistry("runs/model_registry.json")
    
    # ヘルスチェックで生成されたモデルパスを使用
    model_path = "runs/snn_experiment/best_model.pth"
    
    # モデルファイルがない場合はダミー作成（エラー回避）
    if not os.path.exists(model_path):
        print(f"⚠️ {model_path} が見つかりません。ダミーファイルを作成します。")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(b"dummy_weights")

    # エキスパートの共通設定 (Smallモデル相当)
    expert_config = {
        "architecture_type": "predictive_coding",
        "d_model": 128,
        "d_state": 64,
        "num_layers": 4,
        "time_steps": 16,
        "n_head": 2,
        "neuron": {"type": "lif"}
    }

    print("🧪 デモ用エキスパートを登録中...")

    # 1. 科学 (Science) エキスパート
    await registry.register_model(
        model_id="science_expert_v1",
        task_description="science", # 検索キーワード
        metrics={"accuracy": 0.95}, # 高い精度を偽装
        model_path=model_path,
        config=expert_config
    )
    print("  - Registered: Science Expert")

    # 2. 歴史 (History) エキスパート
    await registry.register_model(
        model_id="history_expert_v1",
        task_description="history",
        metrics={"accuracy": 0.92},
        model_path=model_path,
        config=expert_config
    )
    print("  - Registered: History Expert")
    
    print("✅ 登録完了。FrankenMoEの構築が可能です。")

if __name__ == "__main__":
    asyncio.run(main())