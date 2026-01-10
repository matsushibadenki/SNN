# ファイルパス: snn_research/evolution/structural_plasticity.py
# 日本語タイトル: Structural Plasticity Engine (Synaptic Rewiring)
# 目的・内容:
#   ROADMAP Phase 3.2 "Self-Evolution" 対応。
#   学習が停滞した際や睡眠中に、不要なシナプスを削除(Pruning)し、
#   新しいシナプスをランダムに生成(Growth)することで、脳の構造を最適化する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class StructuralPlasticity(nn.Module):
    """
    ニューラルネットワークの構造的可塑性を管理するクラス。
    Synaptic Pruning (刈り込み) と Synaptic Growth (新生) を行う。
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = {}):
        super().__init__()
        self.model = model
        self.pruning_rate = config.get("pruning_rate", 0.1) # 下位10%を削除
        self.growth_rate = config.get("growth_rate", 0.1)   # 同数を新生
        self.noise_std = config.get("noise_std", 0.01)      # 新生時の初期化ノイズ
        
        logger.info("🧬 Structural Plasticity Engine initialized.")

    def evolve_structure(self) -> Dict[str, int]:
        """
        構造を進化させる（Re-wiring）。
        重みの絶対値が小さい接続を削除し、ランダムな新しい接続（重み）を追加する。
        """
        total_pruned = 0
        total_grown = 0
        
        # 線形層(Linear)のみを対象とする（CNN等は構造固定が一般的）
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # マスク処理ではなく、直接重みを書き換える簡易実装
                # (SNNやスパースモデルではマスクが一般的だが、ここではDenseモデルの再初期化として扱う)
                
                with torch.no_grad():
                    weights = module.weight.data
                    
                    # 1. 重要度判定 (Magnitude-based)
                    importance = weights.abs()
                    threshold = torch.quantile(importance, self.pruning_rate)
                    
                    # 2. Pruning (マスク作成: 閾値以下を0にする = 刈り込み)
                    mask = importance > threshold
                    
                    # 実際に0にする（接続を切る）
                    pruned_weights = weights * mask
                    num_pruned = (weights.numel() - mask.sum()).item()
                    
                    # 3. Growth (Re-genesis)
                    # 刈り込まれた部分（0になった部分）に新しいランダムな値を入れる
                    # これにより「死んだ接続」が「新しい可能性」として復活する
                    
                    # 0の部分を特定
                    dead_mask = ~mask
                    
                    # 新しい重みを生成 (He initialization scale or small noise)
                    new_connections = torch.randn_like(weights) * self.noise_std
                    
                    # 既存の有効な重み + (死んだ場所 * 新しい重み)
                    final_weights = pruned_weights + (new_connections * dead_mask.float())
                    
                    # 更新
                    module.weight.data = final_weights
                    
                    total_pruned += num_pruned
                    total_grown += num_pruned # Pruneした分だけGrowする (固定密度)
                    
        logger.info(f"🧬 Evolved Structure: Pruned & Regrown {total_grown} synapses.")
        
        return {
            "pruned": total_pruned,
            "grown": total_grown
        }

    def inject_noise(self, intensity: float = 0.01):
        """
        全体に微小なノイズを加えて局所解からの脱出を促す（Perturbation）。
        """
        for param in self.model.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    noise = torch.randn_like(param) * intensity
                    param.add_(noise)
        logger.info(f"💉 Injected synaptic noise (std={intensity})")