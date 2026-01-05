# ファイルパス: snn_research/agent/digital_life_form.py
# Title: Digital Life Form (Type Safe)
# Description: "Tensor not callable" エラー修正済み。update_drivesを使用。

import logging
import random
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING, cast

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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DigitalLifeForm:
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
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None}
        self.ethical_constraints: List[str] = [
            "harm", "deceive", "error", "illegal"]

    def start(self):
        self.running = True
        logging.info("🚀 DigitalLifeForm activated.")
        asyncio.create_task(self.life_cycle())

    def stop(self):
        self.running = False
        logging.info("🛑 DigitalLifeForm deactivating.")

    async def life_cycle(self):
        while self.running:
            try:
                await self.life_cycle_step()
            except Exception as e:
                logging.error(
                    f"Critical error in life cycle: {e}", exc_info=True)
            await asyncio.sleep(5)

    async def life_cycle_step(self):
        logging.info("\n--- 🧠 New Cognitive Cycle ---")

        self._handle_causal_credit()

        # 状態評価
        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()

        await self._update_ethical_preferences()

        goal = self._formulate_goal(internal_state, performance_eval)
        logging.info(f"🎯 New Goal: {goal}")

        plan = await self.planner.create_plan(goal)
        if not plan.task_list:
            plan.task_list = [
                {"task": "perform_active_inference", "description": "Explore environment"}]

        for task in plan.task_list:
            action = task.get('task')
            if not action:
                continue

            if not await self._check_action_ethics(action):
                continue

            logging.info(f"▶️ Executing task: {action}")
            result, reward, expert_used = await self._execute_action(action, internal_state, performance_eval)

            if isinstance(result, dict):
                self.symbol_grounding.process_observation(
                    result, context=f"action '{action}'")

            decision_context = {"goal": goal,
                                "performance_eval": performance_eval}
            self.memory.record_experience(self.state, action, result, {
                                          "external": reward}, expert_used, decision_context)

            self._update_motivation(reward)

            self.state["last_action"] = action
            self.state["last_result"] = result

            if reward < 0 or performance_eval.get("status") in ["knowledge_gap", "capability_gap"]:
                logging.info("🧬 Triggering self-evolution...")
                await self.self_evolving_agent.evolve(performance_eval, internal_state)
                break

    def _update_motivation(self, reward: float):
        # 予測誤差と成功率を計算（簡易版）
        surprise = random.random() * 0.5
        task_success = True if reward > 0 else False

        # 修正: 明示的にキャストし、存在するメソッド(update_drives)を呼び出す
        motivation_sys = cast(IntrinsicMotivationSystem,
                              self.motivation_system)

        # update_drives(surprise, energy, fatigue, success)
        # energy, fatigueは簡易的に乱数または固定値
        motivation_sys.update_drives(
            surprise=surprise,
            energy_level=random.random() * 100,
            fatigue_level=random.random() * 0.1,
            task_success=task_success
        )

    async def _update_ethical_preferences(self):
        if not hasattr(self.memory, 'rag_system'):
            return
        # (省略: RAGロジック)

    async def _check_action_ethics(self, action: str) -> bool:
        for constraint in self.ethical_constraints:
            if constraint in action.lower():
                return False
        return True

    def _formulate_goal(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        if performance_eval.get("status") == "capability_gap":
            return "Evolve architecture."
        return "Minimize expected free energy."

    def _handle_causal_credit(self):
        pass

    async def _execute_action(self, action: str, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> tuple[Dict[str, Any], float, List[str]]:
        return {"status": "executed"}, 0.5, []
