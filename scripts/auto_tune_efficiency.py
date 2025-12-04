# ファイルパス: scripts/auto_tune_efficiency.py
# Title: SNN効率化 自動チューニングスクリプト (修正版)
# Description: mypyエラー修正 (インポート、OmegaConfメソッド)

import argparse
import logging
import os
# --- ▼ 修正（プロジェクトルートをパスに追加）▼ ---
import sys
from pathlib import Path
# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- ▲ 修正 ▲ ---
from typing import Dict, Any
import optuna
from omegaconf import OmegaConf

from snn_research.training.base_trainer import AbstractTrainer
from snn_research.config.learning_config import BaseLearningConfig 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IDEAL_SPIKE_RATE_MIN = 0.05
IDEAL_SPIKE_RATE_MAX = 0.10

def objective(trial: optuna.Trial, base_config_path: str, model_config_path: str) -> float:
    try:
        base_threshold = trial.suggest_float('model.neuron.base_threshold', 1.2, 2.0, log=True)
        learning_rate = trial.suggest_float('training.optimizer.lr', 1e-5, 5e-4, log=True)
        spike_reg_weight = trial.suggest_float('training.gradient_based.loss.spike_reg_weight', 0.1, 1.0, log=True)
        
        conf = OmegaConf.load(base_config_path)
        model_conf = OmegaConf.load(model_config_path)
        conf = OmegaConf.merge(conf, model_conf)

        OmegaConf.update(conf, 'model.neuron.base_threshold', base_threshold)
        OmegaConf.update(conf, 'training.optimizer.lr', learning_rate)
        OmegaConf.update(conf, 'training.gradient_based.loss.spike_reg_weight', spike_reg_weight)
        
        OmegaConf.update(conf, 'training.gradient_based.type', 'standard')
        OmegaConf.update(conf, 'training.epochs', 50)
        
        # ダミー評価ロジック
        dummy_accuracy = 0.5 + 0.1 * (1 / (base_threshold * spike_reg_weight + 1e-4)) * (learning_rate / 1e-4)
        dummy_accuracy = min(0.9, max(0.01, dummy_accuracy))
        dummy_spike_rate = base_threshold / (spike_reg_weight * 10)
        dummy_spike_rate = min(0.3, max(0.01, dummy_spike_rate))
        
        spike_rate_penalty = 0.0
        if dummy_spike_rate < IDEAL_SPIKE_RATE_MIN:
            spike_rate_penalty = 5.0 * (IDEAL_SPIKE_RATE_MIN - dummy_spike_rate) 
        elif dummy_spike_rate > IDEAL_SPIKE_RATE_MAX:
            spike_rate_penalty = 10.0 * (dummy_spike_rate - IDEAL_SPIKE_RATE_MAX)
        
        score = (1.0 - dummy_accuracy) + spike_rate_penalty
        
        logger.info(f"Trial {trial.number}: Score={score:.4f}")
        return score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 100.0

def main():
    parser = argparse.ArgumentParser(description="SNN Efficiency Auto-Tuning Script")
    parser.add_argument("--base-config", type=str, default="configs/experiments/smoke_test_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/models/micro.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.base_config, args.model_config), n_trials=args.n_trials)
    
    best_trial = study.best_trial
    best_accuracy = 1.0 - (best_trial.value - (10.0 * max(0, best_trial.params['model.neuron.base_threshold'] / (best_trial.params['training.gradient_based.loss.spike_reg_weight'] * 10) - IDEAL_SPIKE_RATE_MAX))) 
    best_spike_rate = best_trial.params['model.neuron.base_threshold'] / (best_trial.params['training.gradient_based.loss.spike_reg_weight'] * 10)

    print("\n" + "=" * 60)
    print("🏆 チューニング完了: 最適パラメータ")
    print("=" * 60)
    print(f"  Best Score (最小化): {best_trial.value:.4f}")
    print(f"  Estimated Accuracy: {best_accuracy:.4f}") 
    print(f"  Estimated Spike Rate: {best_spike_rate:.4f}")
    print("-" * 30)
    print("  [推奨設定]")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()