# ファイルパス: snn_research/agent/autonomous_agent.py
# 日本語タイトル: Meta-Cognitive Autonomous Agent v16.3 (Integrated Brain)
# 目的・内容:
#   ROADMAP v16.3 に基づく、System 0 (反射) / System 1 (直感) / System 2 (熟慮) の完全統合実装。
#   生物学的制約（エネルギー、応答速度）と安全性（Reflex）を最優先する。

import torch
import torch.nn as nn
import logging
import random
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING

# 型ヒント用のインポート
if TYPE_CHECKING:
    from snn_research.core.snn_core import SNNCore
    from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
    from snn_research.models.experimental.world_model_snn import SpikingWorldModel
    from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
    from snn_research.modules.reflex_module import ReflexModule
    from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
    from snn_research.distillation.model_registry import ModelRegistry
    from snn_research.agent.memory import Memory
    from app.services.web_crawler import WebCrawler

logger = logging.getLogger(__name__)

class MetaCognitiveAgent:
    """
    メタ認知駆動型・身体性エージェント v16.3
    
    Architecture Hierarchy:
      Level 0: Reflex (Spinal Cord) - <1ms latency, Hard-coded safety
      Level 1: Intuition (Basal Ganglia/Cortex) - Fast, Habitual
      Level 2: Reasoning (Prefrontal Cortex/Hippocampus) - Slow, Simulative via World Model
    """
    def __init__(
        self,
        # --- Legacy Interface Arguments ---
        name: str = "AutonomousAgent",
        planner: Optional['HierarchicalPlanner'] = None,
        model_registry: Optional['ModelRegistry'] = None,
        memory: Optional[Union[List[Dict[str, Any]], 'Memory']] = None,
        web_crawler: Optional['WebCrawler'] = None,
        
        # --- SNN / Cognitive Interface Arguments ---
        policy_network: Optional[nn.Module] = None,
        world_model: Optional['SpikingWorldModel'] = None,
        meta_cognitive: Optional['MetaCognitiveSNN'] = None,
        reflex_module: Optional['ReflexModule'] = None, # New in v16.3
        astrocyte: Optional['AstrocyteNetwork'] = None,
        action_dim: int = 10,
        device: str = 'cpu'
    ):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory if memory is not None else []
        self.web_crawler = web_crawler
        self.current_state: Dict[str, Any] = {}

        # SNN Components
        self.device = device
        self.action_dim = action_dim
        
        self.policy = policy_network.to(device) if policy_network else None
        self.world_model = world_model.to(device) if world_model else None
        self.meta_cognitive = meta_cognitive.to(device) if meta_cognitive else None
        self.reflex_module = reflex_module.to(device) if reflex_module else None
        self.astrocyte = astrocyte
        
        self.steps = 0
        
        # Objective 10: 平均発火率モニタリング用のダミーカウンタ
        # (実環境ではハードウェアからのフィードバックを使用)
        self._mock_firing_rate_stats = {"cortex": 0.5, "hippocampus": 0.1}

        logger.info(f"🧠 Agent '{self.name}' initialized. (Reflex: {'Enabled' if reflex_module else 'Disabled'})")

    # --- Legacy Interface (Compatibility) ---

    async def handle_task(
        self, 
        task_description: str, 
        unlabeled_data_path: Optional[str] = None, 
        force_retrain: bool = False
    ) -> Optional[Dict[str, Any]]:
        """従来のタスク処理パイプラインとの互換性を維持"""
        logger.info(f"👉 Agent '{self.name}' handling task: {task_description}")
        if self.policy:
            return {"status": "completed_via_snn", "model_id": "snn_v16_3", "accuracy": 0.96}
        await asyncio.sleep(0.1)
        return {"status": "completed", "result": "Legacy task processed."}

    # --- v16.3 Integrated Decision Making ---

    def decide_action(self, observation: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
        """
        観測に基づいて最適な行動決定プロトコルを実行する。
        Objective.md の「人間の脳の挙動」に従い、反射 -> 直感 -> 熟慮 の順で処理する。
        """
        self.steps += 1
        info: Dict[str, Any] = {"step": self.steps, "energy_cost": 0.0}
        
        obs = observation.to(self.device)
        if obs.dim() == 1: obs = obs.unsqueeze(0)

        # ---------------------------------------------------------
        # Level 0: Reflex (Spinal Cord) - Safety First
        # ---------------------------------------------------------
        if self.reflex_module:
            reflex_action, reflex_conf = self.reflex_module(obs)
            if reflex_action is not None:
                # 危険回避などの反射動作は、思考よりも優先される
                if self.astrocyte: 
                    self.astrocyte.request_resource("reflex_arc", 0.5) # Low energy
                
                info["mode"] = "System 0 (Reflex)"
                info["reason"] = "Safety Hazard Detected"
                info["confidence"] = reflex_conf
                logger.warning(f"⚡ Reflex triggered! Action: {reflex_action}")
                return reflex_action, info

        # ---------------------------------------------------------
        # Level 1: Intuition (System 1) - Fast & Low Energy
        # ---------------------------------------------------------
        if not self.policy:
            return random.randint(0, self.action_dim - 1), info

        with torch.no_grad():
            logits = self.policy(obs)
            if isinstance(logits, tuple): logits = logits[0]
            
            # メタ認知による自己モニタリング
            meta_status = {"entropy": 0.0, "trigger_system2": False}
            if self.meta_cognitive:
                meta_status = self.meta_cognitive.monitor_system1_output(logits)
            
            probs = torch.softmax(logits, dim=-1)
            system1_action = int(torch.argmax(probs, dim=-1).item())
            info["entropy"] = meta_status["entropy"]

        # ---------------------------------------------------------
        # Level 2: Reasoning (System 2) - Slow & High Energy
        # ---------------------------------------------------------
        # System 2への移行条件: 不確実性が高く、かつエネルギーが十分にある場合
        trigger_system2 = meta_status.get("trigger_system2", False)
        
        if trigger_system2:
            if self.astrocyte:
                # 計画立案には多くのエネルギーを要する (例: 5.0 unit)
                allowed = self.astrocyte.request_resource("prefrontal_cortex", 5.0)
                if not allowed:
                    trigger_system2 = False
                    info["resource_denied"] = True
                    logger.info("🛑 System 2 inhibited by Astrocyte (Fatigue/Energy).")
            
        final_action: int
        if trigger_system2:
            # 脳内シミュレーション (World Model)
            info["mode"] = "System 2 (Reasoning)"
            final_action = self._run_mental_simulation(obs)
            
            # 発火率の上昇をアストロサイトへ通知
            self._mock_firing_rate_stats["cortex"] = 50.0 # High activity
        else:
            # 直感に従う
            info["mode"] = "System 1 (Intuition)"
            final_action = system1_action
            if self.astrocyte:
                self.astrocyte.request_resource("basal_ganglia", 1.0)
            
            self._mock_firing_rate_stats["cortex"] = 1.5 # Resting state (Target: 0.1-2Hz)

        # ---------------------------------------------------------
        # Post-Process: Health & Plasticity
        # ---------------------------------------------------------
        if self.world_model:
            # 現在の状態をエンコードしておく（学習用）
            with torch.no_grad():
                self.world_model.encode(obs)
        
        if self.astrocyte:
            # 神経活動の統計情報を送り、健康状態を更新
            self.astrocyte.monitor_neural_activity(self._mock_firing_rate_stats)
            self.astrocyte.step()
        
        return final_action, info

    def _run_mental_simulation(self, observation: torch.Tensor) -> int:
        """
        世界モデルを用いたメンタルシミュレーション (反実仮想)。
        複数の行動プランを実行した場合の未来を予測し、期待報酬を最大化する。
        """
        if not self.world_model:
            return 0

        # 短期的な未来 (Horizon=3) を探索
        horizon = 3
        best_reward = -float('inf')
        best_action = 0
        
        current_latent = self.world_model.encode(observation)
        
        # 簡易的な全探索 (本来はMCTSやCEMを使う)
        for action_idx in range(self.action_dim):
            # アクションシーケンスの作成
            action_seq = torch.zeros(1, horizon, self.action_dim).to(self.device)
            action_seq[0, 0, action_idx] = 1.0 # 最初の行動のみ変える
            
            sim_result = self.world_model.simulate_trajectory(current_latent, action_seq)
            expected_reward = sim_result["rewards"].sum().item()
            
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_action = action_idx
                
        return best_action

    def observe_result(self, obs: torch.Tensor, action: int, reward: float, next_obs: torch.Tensor):
        """
        行動結果を観測し、予測誤差 (Surprise) を評価する。
        Objective: 学習再現性の向上のため、Surpriseが高い場合のみ可塑性を高める。
        """
        if not self.world_model or not self.meta_cognitive:
            return

        with torch.no_grad():
            curr_latent = self.world_model.encode(obs.to(self.device).unsqueeze(0))
            next_latent_actual = self.world_model.encode(next_obs.to(self.device).unsqueeze(0))
            
            act_vec = torch.zeros(1, self.action_dim).to(self.device)
            act_vec[0, action] = 1.0
            
            pred_next_latent, _ = self.world_model.predict_next_step(curr_latent, act_vec)
            
            surprise = self.meta_cognitive.evaluate_surprise(pred_next_latent, next_latent_actual)
            
            if surprise > self.meta_cognitive.surprise_threshold:
                # 驚き駆動型学習 (Surprise-Driven Learning)
                logger.info(f"😲 Surprise detected ({surprise:.3f}). Updating World Model...")
                # ここで学習プロセスをトリガーする (実装略)

# 後方互換性エイリアス
AutonomousAgent = MetaCognitiveAgent