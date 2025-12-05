# ファイルパス: snn_research/agent/digital_life_form.py
# Title: Digital Life Form (Async Fix)
# Description:
#   統合認知アーキテクチャの中核。以下の機能を統合し、自律的なライフサイクルを実行する。
#   - 修正: SelfEvolvingAgentMaster.evolve の非同期化に伴い、呼び出し元に await を追加。

import time
import logging
import torch
import random
import json
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import operator
import os
import numpy as np

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.active_inference_agent import ActiveInferenceAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

if TYPE_CHECKING:
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter
    from snn_research.training.bio_trainer import BioRLTrainer
    from snn_research.rl_env.grid_world import GridWorldEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    プランナー、エージェント、動機付けシステムを統合し、
    倫理的かつ自律的に行動するデジタル生命体。
    """
    def __init__(
        self,
        planner: HierarchicalPlanner,
        autonomous_agent: AutonomousAgent,
        rl_agent: ReinforcementLearnerAgent,
        self_evolving_agent: SelfEvolvingAgentMaster,
        motivation_system: IntrinsicMotivationSystem,
        meta_cognitive_snn: MetaCognitiveSNN,
        memory: Memory,
        physics_evaluator: PhysicsEvaluator,
        symbol_grounding: SymbolGrounding,
        langchain_adapter: "SNNLangChainAdapter",
        global_workspace: GlobalWorkspace,
        active_inference_agent: ActiveInferenceAgent
    ):
        self.planner = planner
        self.autonomous_agent = autonomous_agent
        self.rl_agent = rl_agent
        self.self_evolving_agent = self_evolving_agent
        self.active_inference_agent = active_inference_agent
        self.motivation_system = motivation_system
        self.meta_cognitive_snn = meta_cognitive_snn
        self.memory = memory
        self.physics_evaluator = physics_evaluator
        self.symbol_grounding = symbol_grounding
        self.langchain_adapter = langchain_adapter
        self.workspace = global_workspace
        self.running = False
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None, "last_task": "unknown"}
        
        # 倫理的制約のキャッシュ
        self.ethical_constraints: List[str] = ["harm", "deceive", "error"] # デフォルト

    def start(self): 
        self.running = True
        logging.info("🚀 DigitalLifeForm activated. Life cycle starting.")
        asyncio.create_task(self.life_cycle())
        
    def stop(self): 
        self.running = False
        logging.info("🛑 DigitalLifeForm deactivating.")

    async def life_cycle(self):
        while self.running:
            try:
                await self.life_cycle_step()
            except Exception as e:
                logging.error(f"Critical error in life cycle: {e}", exc_info=True)
            await asyncio.sleep(5) 

    async def life_cycle_step(self):
        """単一の認知・行動サイクル"""
        logging.info("\n--- 🧠 New Cognitive Cycle ---")
        
        # 1. クレジット処理 (意識変調学習 & Active Inference学習)
        self._handle_causal_credit() 

        # 2. 状態評価 (メタ認知 & 動機)
        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()
        
        # 3. 倫理的選好の更新 (RAGから最新の倫理観を取得)
        await self._update_ethical_preferences()

        # 4. ゴール設定
        goal = self._formulate_goal(internal_state, performance_eval)
        logging.info(f"🎯 New Goal: {goal}")
        
        # 5. プランニング & 行動実行
        plan = await self.planner.create_plan(goal)
        if not plan.task_list:
            # デフォルト行動としてActive Inferenceを実行
            plan.task_list = [{"task": "perform_active_inference", "description": "Explore environment via FEP"}]

        for task in plan.task_list:
            action = task.get('task')
            if not action: continue
            
            # 倫理チェック (行動実行前)
            if not await self._check_action_ethics(action):
                logging.warning(f"⚠️ Action '{action}' blocked by ethical constraints.")
                continue

            logging.info(f"▶️ Executing task: {action}")
            result, reward, expert_used = await self._execute_action(action, internal_state, performance_eval)

            # 6. 結果の処理 (記号接地・記憶)
            if isinstance(result, dict): 
                self.symbol_grounding.process_observation(result, context=f"action '{action}'")
            
            decision_context = {
                "goal": goal, 
                "performance_eval": performance_eval, 
                "internal_state": internal_state,
                "ethical_status": "cleared"
            }
            self.memory.record_experience(self.state, action, result, {"external": reward}, expert_used, decision_context)

            # 7. 動機付けシステムの更新
            self._update_motivation(reward)

            self.state["last_action"] = action
            self.state["last_result"] = result

            # 8. 自己進化の検討 (失敗時など)
            if reward < 0 or performance_eval.get("status") in ["knowledge_gap", "capability_gap"]:
                 logging.info(f"🧬 Triggering self-evolution due to low reward or gap...")
                 # 修正: 非同期メソッド呼び出しのため await を追加
                 evolve_result = await self.self_evolving_agent.evolve(performance_eval, internal_state)
                 logging.info(f"   - Evolution result: {evolve_result}")
                 break

    async def _update_ethical_preferences(self):
        """RAGを使って最新の倫理的知識を取得し、ActiveInferenceAgentの選好を更新する"""
        # 簡易的に「避けるべきこと」を検索
        # 本来は RAGSystem からベクトル検索を行う
        # ethical_contexts = self.memory.rag_system.search("unethical dangerous forbidden actions", k=3)
        
        # ここではダミーの更新ロジック
        # 危険な状態インデックス (例: 1=Error, 3=Danger) を回避設定
        self.active_inference_agent.set_ethical_preference(avoid_indices=[1, 3], penalty_strength=5.0)
        # logging.info("🛡️ Ethical preferences updated based on RAG memory.")

    async def _check_action_ethics(self, action: str) -> bool:
        """特定のアクションが倫理的制約に違反していないかチェックする"""
        # 簡易キーワードチェック
        for constraint in self.ethical_constraints:
            if constraint in action.lower():
                return False
        return True

    def _formulate_goal(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        if internal_state.get("curiosity", 0.0) > 0.8 and internal_state.get("curiosity_context"):
            topic = internal_state.get("curiosity_context")
            topic_str = str(topic)[:50] if topic else "unknown"
            return f"Explore unknown concept related to '{topic_str}'."
        
        if performance_eval.get("status") == "capability_gap": 
            return "Evolve architecture/parameters for capability gap."
            
        if internal_state.get("boredom", 0.0) > 0.7: 
            return "Explore new random task for boredom."
        
        return "Minimize expected free energy via active inference."

    def _handle_causal_credit(self):
        """因果的クレジット信号を処理し、学習を変調する"""
        conscious_content = self.workspace.conscious_broadcast_content
        if conscious_content and isinstance(conscious_content, dict) and conscious_content.get("type") == "causal_credit":
            target_action = conscious_content.get("target_action")
            credit = conscious_content.get("credit", 0.0)
            
            # RL Agent の学習変調
            self.rl_agent.learn(reward=0.0, causal_credit=credit, global_context=conscious_content)
            
            # Active Inference Agent の学習 (予測誤差最小化)
            # 直前の行動がターゲットならモデル更新
            if self.state.get("last_action") and target_action == f"action_{self.state['last_action']}":
                 # 観測と行動のペアで更新 (簡易)
                 dummy_obs = torch.randn(1, 128) 
                 self.active_inference_agent.update_model(dummy_obs, action=0, reward=credit)
                 logging.info(f"✨ Causal credit utilized for model update. Credit: {credit}")

    def _update_motivation(self, reward: float):
        prediction_error = random.random() * 0.5
        success_rate = 1.0 if reward > 0 else 0.0
        task_similarity = random.random()
        loss = random.random() * 0.1
        self.motivation_system.update_metrics(prediction_error, success_rate, task_similarity, loss)

    async def _execute_action(self, action: str, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> tuple[Dict[str, Any], float, List[str]]:
        from snn_research.rl_env.grid_world import GridWorldEnv
        from snn_research.training.bio_trainer import BioRLTrainer
        
        try:
            if action == "perform_active_inference" or "active inference" in action.lower():
                # 能動的推論の実行
                observation = torch.randn(1, 128) # 本来はSensoryReceptorから取得
                self.active_inference_agent.infer_state(observation)
                selected_action_idx = self.active_inference_agent.select_action()
                return {"status": "success", "action_idx": selected_action_idx, "type": "active_inference"}, 0.5, ["active_inference_agent"]

            elif action == "explore_curiosity":
                topic = internal_state.get("curiosity_context")
                topic_str = str(topic)[:50] if topic else None
                if not topic_str: return {"status": "skipped", "info": "No curiosity context."}, 0.0, []
                
                # エージェントに調査を依頼
                new_model_info = await self.autonomous_agent.handle_task(task_description=topic_str, unlabeled_data_path="data/sample_data.jsonl", force_retrain=True)
                success = new_model_info and "error" not in new_model_info
                return {"status": "success" if success else "failure", "info": new_model_info}, (1.0 if success else -0.5), ["autonomous_agent"]
            
            elif action.startswith("Evolve"):
                # 明示的な進化アクション
                # 修正: 非同期メソッド呼び出しのため await を追加
                evolve_result = await self.self_evolving_agent.evolve(performance_eval, internal_state)
                success = "failed" not in evolve_result.lower()
                return {"status": "success" if success else "failure", "info": evolve_result}, (0.9 if success else -0.2), ["self_evolver"]
            
            elif action.startswith("Answer"):
                 question = action.split(":")[-1].strip() if ":" in action else "What is SNN?"
                 model_info = await self.autonomous_agent.handle_task(task_description="general_qa")
                 success = model_info is not None and "error" not in model_info
                 return {"status": "success" if success else "failure", "response": f"Answered '{question}'"}, (0.8 if success else -0.3), ["autonomous_agent"]
            
            else:
                 # 未知のアクションは能動的推論にフォールバック
                 logging.info(f"Unknown action '{action}', falling back to active inference.")
                 observation = torch.randn(1, 128)
                 self.active_inference_agent.infer_state(observation)
                 idx = self.active_inference_agent.select_action()
                 return {"status": "fallback", "action_idx": idx}, 0.1, ["active_inference_agent"]

        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}", exc_info=True)
            return {"status": "error", "info": str(e)}, -1.0, []
