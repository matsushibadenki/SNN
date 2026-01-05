# ファイルパス: snn_research/optimization/hseo.py
# (修正)
# 修正: 循環インポートエラーを解消するため、TrainingContainer のインポートを
#       トップレベルから evaluate_snn_params 関数内に移動。

import numpy as np
import torch
import sys
import logging
import os
from typing import List, Tuple, Callable, Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
# --- ▼ 修正: ここでのインポートを削除 ▼ ---
# from app.containers import TrainingContainer
# --- ▲ 修正 ▲ ---


# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトルートをPythonパスに追加 (train.py を呼び出すため)
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 1. SNN評価関数 (目的関数) ---

def evaluate_snn_params(
    model_config_path: str,
    base_training_config_path: str,
    params_to_override: Dict[str, Any],
    eval_epochs: int = 1, # (HSEOではエピソード数として解釈)
    device: str = "cpu",
    task_name: str = "GridWorld", # (HSEOはRLタスクで評価)
    metric_to_optimize: str = "reward" # "reward" (最大化) または "loss" (最小化)
) -> float:
    """
    (改善 v2) 指定されたパラメータで「微分不要な」SNNモデルを短期間訓練し、
    性能メトリクス（報酬）を返す。
    BioRLTrainer (GridWorld) を直接実行する。

    Args:
        model_config_path (str): モデル設定ファイルのパス (BioSNN用ではないが互換性のため)。
        base_training_config_path (str): 基本訓練設定ファイルのパス。
        params_to_override (Dict[str, Any]): 上書きするパラメータ (例: {"training.biologically_plausible.causal_trace.learning_rate": 0.01})。
        eval_epochs (int): 評価のために実行するエピソード数。
        metric_to_optimize (str): "reward" (最大化) または "loss" (最小化)。

    Returns:
        float: 評価スコア。
               (Optuna/HSEOは最小化を目指すため、reward の場合は -reward を返す)
    """
    # --- ▼ 修正: ここでインポートする (循環参照回避) ▼ ---
    from app.containers import TrainingContainer
    # --- ▲ 修正 ▲ ---
    
    logger.info(f"HSEO: Evaluating BioRL parameters (Episodes: {eval_epochs})...")
    logger.info(f"HSEO: Overrides: {params_to_override}")

    try:
        # --- 1. DIコンテナと設定のロード ---
        container = TrainingContainer()
        container.config.from_yaml(base_training_config_path)
        # (BioSNNはモデルコンフィグを使わないが、念のためロード)
        if model_config_path and os.path.exists(model_config_path):
            container.config.from_yaml(model_config_path)
        
        # --- 2. パラメータの上書き ---
        cfg: DictConfig = container.config()
        for key, value in params_to_override.items():
            OmegaConf.update(cfg, key, value, merge=True)
            
        # HSEOは生物学的学習則のパラメータを最適化すると仮定
        cfg.training.paradigm.from_value("bio-causal-sparse") # 仮にCausalTraceを使用
        
        # --- 3. BioRLTrainer の実行 ---
        # (DIコンテナが設定に基づき、CausalTrace V2 などの学習則を持つ
        #  エージェントとトレーナーを構築する)
        trainer = container.bio_rl_trainer()
        
        # 訓練（評価）の実行
        results: Dict[str, float] = trainer.train(num_episodes=eval_epochs)
        
        final_reward: float = results.get('final_average_reward', 0.0)
        
        # --- 4. メトリクスの返却 ---
        metric_value: float
        if metric_to_optimize == "reward":
            # 報酬は最大化を目指すので、最小化のために負の値を返す
            metric_value = -final_reward
        else:
            # (もし損失を返すロジックがあれば)
            metric_value = results.get('final_loss', float('inf'))

        logger.info(f"HSEO: Evaluation complete. Reward: {final_reward:.4f}, Metric ({-final_reward:.4f})")
        return metric_value

    except Exception as e:
        logger.error(f"HSEO: BioRLTrainer evaluation failed: {e}", exc_info=True)
        return float('inf') # 失敗時は最悪のスコア（無限大）を返す


# --- 2. HSEO (PSO) コアアルゴリズム ---

def optimize_with_hseo(
    objective_function: Callable[[np.ndarray], np.ndarray],
    dim: int,
    num_particles: int,
    max_iterations: int,
    exploration_range: List[Tuple[float, float]], # パラメータごとの [min, max] のリスト
    w: float = 0.5,  # 慣性係数
    c1: float = 1.5, # 個体最適解への引力
    c2: float = 1.5, # 全体最適解への引力
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    粒子群最適化 (PSO) を実行する。
    HSEO (Hybrid Swarm) の実装としてPSOを使用する。

    Args:
        objective_function (Callable): 目的関数。パーティクル集団(N, D)を受け取り、スコア配列(N,)を返す。
        dim (int): パラメータの次元数。
        num_particles (int): パーティクルの数。
        max_iterations (int): 最大イテレーション数。
        exploration_range (List[Tuple[float, float]]): 各次元の探索範囲 [min, max] のリスト。
        w (float): 慣性係数。
        c1 (float): 個体最適解への引力。
        c2 (float): 全体最適解への引力。
        seed (Optional[int]): 乱数シード。
        verbose (bool): ログ出力の有無。

    Returns:
        Tuple[np.ndarray, float]: (最適パラメータ, 最適スコア)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 探索範囲をNumPy配列に変換
    min_bounds = np.array([r[0] for r in exploration_range])
    max_bounds = np.array([r[1] for r in exploration_range])
    range_width = max_bounds - min_bounds

    # 1. 初期化
    # パーティクルの位置 (パラメータセット)
    particles_pos = min_bounds + np.random.rand(num_particles, dim) * range_width
    # パーティクルの速度
    particles_vel = np.random.randn(num_particles, dim) * 0.1
    
    # 各パーティクルのパーソナルベスト (pbest)
    pbest_pos = particles_pos.copy()
    pbest_scores = np.full(num_particles, float('inf'))
    
    # グローバルベスト (gbest)
    gbest_pos = np.zeros(dim)
    gbest_score = float('inf')

    if verbose:
        logger.info(f"HSEO (PSO): Starting optimization with {num_particles} particles for {max_iterations} iterations.")
        logger.info(f"HSEO (PSO): Dimension={dim}, Bounds={exploration_range}")

    # 2. イテレーション
    for i in range(max_iterations):
        # 目的関数で全パーティクルのスコアを評価
        scores = objective_function(particles_pos)
        
        # pbest の更新
        update_mask = scores < pbest_scores
        pbest_pos[update_mask] = particles_pos[update_mask]
        pbest_scores[update_mask] = scores[update_mask]
        
        # gbest の更新
        best_particle_idx = np.argmin(scores)
        if scores[best_particle_idx] < gbest_score:
            gbest_score = scores[best_particle_idx]
            gbest_pos = particles_pos[best_particle_idx].copy()
            
            if verbose:
                logger.info(f"HSEO (PSO) [Iter {i+1}/{max_iterations}]: New gbest score = {gbest_score:.6f}")
        
        # 3. 速度と位置の更新
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        
        # PSO更新式
        cognitive_vel = c1 * r1 * (pbest_pos - particles_pos)
        social_vel = c2 * r2 * (gbest_pos - particles_pos)
        particles_vel = w * particles_vel + cognitive_vel + social_vel
        
        particles_pos = particles_pos + particles_vel
        
        # 探索範囲内にクリッピング
        particles_pos = np.clip(particles_pos, min_bounds, max_bounds)

    if verbose:
        logger.info(f"HSEO (PSO): Optimization finished. Final best score = {gbest_score:.6f}")
        logger.info(f"HSEO (PSO): Best parameters = {gbest_pos}")

    return gbest_pos, gbest_score
