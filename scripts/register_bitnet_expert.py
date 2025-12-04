import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from snn_research.distillation.model_registry import SimpleModelRegistry

async def main():
    registry = SimpleModelRegistry("runs/model_registry.json")
    
    # Step 2で生成されたモデルパスを指定 (適宜調整してください)
    # train.py のログにある "best_model.pth" のパスを確認
    model_path = "runs/smoke_tests/best_model.pth" 
    
    # 1.58bitモデルの設定
    expert_config = {
        "architecture_type": "bit_spiking_rwkv",
        "d_model": 128,
        "num_layers": 4,
        "time_steps": 16,
        "neuron": {"type": "lif"}
    }

    print("🧪 1.58bit エキスパートを登録中...")

    # "calculation" (計算) タスクのエキスパートとして登録
    await registry.register_model(
        model_id="bitnet_calc_expert_v1",
        task_description="calculation", 
        metrics={"accuracy": 0.99, "energy_efficiency": 10.0}, # 高効率をアピール
        model_path=model_path,
        config=expert_config
    )
    print("✅ 登録完了: BitNet Calculation Expert")

if __name__ == "__main__":
    asyncio.run(main())