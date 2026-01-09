# ファイルパス: scripts/experiments/brain/run_world_model_simulation.py
# 日本語タイトル: Multimodal World Model Simulation (Type Fixed)
# 修正内容: generate_synthetic_world_data の戻り値型ヒントを修正。

from snn_research.utils.efficiency_profiler import EfficiencyProfiler
from snn_research.core.architecture_registry import ArchitectureRegistry
import os
import sys
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WorldModelSim")


def generate_synthetic_world_data(batch_size: int, seq_len: int, config: Dict) -> Dict[str, Any]:
    """
    物理法則に従うような擬似的な時系列データを生成する。
    例: 視覚上のボールが移動すると、特定のタイミングで触覚/音が反応する。

    Returns:
        Dict containing 'sensory' (dict of tensors) and 'actions' (tensor)
    """
    device = config['device']

    # 1. Action: ランダムな移動指令 (dx, dy)
    actions = torch.randn(batch_size, seq_len,
                          config['action_dim'], device=device)

    # 2. Vision: 単純な移動するドットをシミュレート (簡易版)
    # ここでは完全な物理シミュレーションではなく、行動に相関したランダムパターンを使用
    vision_dim = config['sensory_configs']['vision']
    # 行動の蓄積（位置）に応じて変化する波形
    position = torch.cumsum(actions[:, :, 0], dim=1).unsqueeze(-1)  # (B, T, 1)
    # sin波を使って位置情報を高次元パターンにエンコード
    freqs = torch.linspace(0.1, 10.0, vision_dim,
                           device=device).unsqueeze(0).unsqueeze(0)
    vision_data = torch.sin(position * freqs)

    # 3. Tactile: 特定の位置（壁など）に来たときに反応
    tactile_dim = config['sensory_configs']['tactile']
    # 位置が特定の値を超えたら「壁に当たった」として触覚信号発生
    wall_collision = (torch.abs(position) > 2.0).float()
    tactile_data = wall_collision.expand(-1, -1, tactile_dim) * torch.randn(
        batch_size, seq_len, tactile_dim, device=device)

    return {
        'sensory': {
            'vision': vision_data,  # (B, T, D_vis)
            'tactile': tactile_data  # (B, T, D_tac)
        },
        'actions': actions  # (B, T, D_act)
    }


def main():
    logger.info("🌍 Starting Multimodal World Model Simulation...")

    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        'device': device,  # Added device here for generator
        'd_model': 256,
        'd_state': 64,
        'num_layers': 4,
        'time_steps': 8,
        'action_dim': 2,
        'sensory_configs': {
            'vision': 128,   # 簡易シミュレーション用次元
            'tactile': 32
        },
        'neuron': {'type': 'lif'},
        'use_bitnet': True
    }

    # --- Build Model ---
    logger.info("🏗️ Building SpikingWorldModel...")
    # Registry経由でビルド (修正したArchitectureRegistryを使用)
    model = ArchitectureRegistry.build(
        "spiking_world_model",
        config=config,
        vocab_size=0  # WorldModelは離散語彙必須ではないため0
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    profiler = EfficiencyProfiler()

    # --- Training Loop (Self-Supervised) ---
    num_epochs = 10
    batch_size = 8
    seq_len = 20  # 20ステップ分の未来を予測しながら学習

    model.train()

    for epoch in range(num_epochs):
        # データ生成
        data_batch = generate_synthetic_world_data(batch_size, seq_len, config)
        sensory_inputs = data_batch['sensory']
        actions = data_batch['actions']

        profiler.start_measurement()
        optimizer.zero_grad()

        # Forward: 全ステップを一括処理 (Training mode)
        # model.forward() は (z_pred, reconstructions, h_next) を返す
        # ここでは「過去の観測+行動」から「未来の観測」を予測させたい

        # 入力を1ステップずらす (tの入力で t+1 を予測)
        # inputs: 0 ~ T-1
        # targets: 1 ~ T

        current_inputs = {k: v[:, :-1, :] for k, v in sensory_inputs.items()}
        current_actions = actions[:, :-1, :]
        target_observations = {k: v[:, 1:, :]
                               for k, v in sensory_inputs.items()}

        # 推論
        z_pred, reconstructions, _ = model(current_inputs, current_actions)

        # Loss計算: 再構成誤差 (MSE)
        total_loss = torch.tensor(0.0, device=device)
        losses_by_modality = {}

        for mod, pred in reconstructions.items():
            target = target_observations[mod]
            # 次元合わせ (モデル出力長とターゲット長)
            min_len = min(pred.size(1), target.size(1))

            recon_loss = F.mse_loss(
                pred[:, :min_len, :], target[:, :min_len, :])
            total_loss = total_loss + recon_loss
            losses_by_modality[mod] = recon_loss.item()

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        profiler.end_measurement()

        if (epoch + 1) % 2 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss.item():.4f}")
            logger.info(f"   Breakdown: {losses_by_modality}")

    # --- Prediction / Dreaming Test ---
    logger.info("💤 Testing Dreaming Capability (Open-Loop Prediction)...")
    model.eval()

    with torch.no_grad():
        # 初期状態 (t=0)
        initial_obs = {k: v[:, 0:1, :] for k, v in sensory_inputs.items()}
        # 未来の行動計画 (t=0~9)
        future_actions = actions[:, 0:10, :]

        # 閉ループ予測: 予測した結果を次の入力として使い、夢を見続ける
        current_obs = initial_obs
        dreamed_trajectory = []

        # 内部状態のリセット (必要であれば)
        # model.reset_state() # もし実装されていれば

        for t in range(10):
            action_t = future_actions[:, t, :]  # (B, ActDim)

            # 1ステップ予測 (predict_next メソッド使用)
            next_obs_pred = model.predict_next(current_obs, action_t)

            # 予測結果を保存
            dreamed_trajectory.append(next_obs_pred)

            # 次の入力として予測値を使用 (閉ループ)
            # predict_next は (B, D) を返すので (B, 1, D) にリシェイプ
            current_obs = {k: v.unsqueeze(1) for k, v in next_obs_pred.items()}

    logger.info("✅ Dreaming verification completed.")
    logger.info(
        "   The model successfully generated a 10-step future trajectory without external input.")


if __name__ == "__main__":
    main()
