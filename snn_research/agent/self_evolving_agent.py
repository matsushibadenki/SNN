# ファイルパス: snn_research/agent/self_evolving_agent.py
# Title: Master Self-Evolving Agent (Fixed Init)
# Description: 親クラス(AutonomousAgent)の初期化シグネチャ変更に対応し、super().__init__呼び出しを修正。

from typing import Dict, Any, Optional, List, Tuple, Callable
import os
import random
import math
import asyncio
import logging
import torch
from omegaconf import OmegaConf
from datetime import datetime

from .autonomous_agent import AutonomousAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from app.services.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem

# ハードウェア連携
from snn_research.hardware.compiler import NeuromorphicCompiler
from snn_research.core.snn_core import SNNCore

HSEO_AVAILABLE = False
try:
    HSEO_AVAILABLE = True
except ImportError:
    logging.warning(
        "⚠️ HSEO module not found. 'hseo_optimize_lp' operator will be disabled.")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SelfEvolvingAgentMaster(AutonomousAgent):
    """
    Phase 5: ニューロシンボリック進化を司るマスターエージェント。
    """
    memory: AgentMemory
    model_registry: ModelRegistry

    def __init__(
        self,
        name: str,
        planner: HierarchicalPlanner,
        model_registry: ModelRegistry,
        memory: AgentMemory,
        web_crawler: WebCrawler,
        meta_cognitive_snn: MetaCognitiveSNN,
        motivation_system: IntrinsicMotivationSystem,
        evolution_threshold: float = 0.5,
        project_root: str = ".",
        model_config_path: Optional[str] = None,
        training_config_path: Optional[str] = None,
        evolution_history_buffer_size: int = 50,
        evolution_learning_rate: float = 0.1,
        evolution_budget: float = 10.0,
        social_learning_probability: float = 0.2
    ):
        # 修正: AutonomousAgentの__init__シグネチャ (input_size, output_size, device, config_path) に合わせる
        # 引数で渡されていないため、適切なデフォルト値を設定
        default_input_size = 64
        default_output_size = 10
        default_device = "cpu"  # 必要であれば引数に追加するか、環境から取得
        if torch.cuda.is_available():
            default_device = "cuda"
        elif torch.backends.mps.is_available():
            default_device = "mps"

        # 親クラスの初期化 (Brain/SNNの構築)
        super().__init__(
            input_size=default_input_size,
            output_size=default_output_size,
            device=default_device,
            config_path=model_config_path
        )

        # 親クラスで管理されない属性を自身で保持
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler  # 個別に設定（親クラスのcrawlerとは別管理または上書き）

        self.meta_cognitive_snn = meta_cognitive_snn
        self.motivation_system = motivation_system
        self.evolution_threshold = evolution_threshold
        self.project_root = project_root

        self.model_config_path = model_config_path or "configs/models/small.yaml"
        self.training_config_path = training_config_path or "configs/templates/base_config.yaml"

        self.evolution_history_buffer_size = evolution_history_buffer_size
        self.evolution_learning_rate = evolution_learning_rate
        self.evolution_budget = evolution_budget
        self.social_learning_probability = social_learning_probability

        self.evolved_config_dir = os.path.join(
            "workspace", "runs", "evolved_configs")
        os.makedirs(self.evolved_config_dir, exist_ok=True)

        # 進化オペレータの定義
        self.evolution_operators: Dict[str, Dict[str, Any]] = {
            "architecture_large": {"func": self._evolve_architecture, "cost": 5.0, "params": {"scale_factor_range": (1.3, 1.8), "layer_increase_range": (2, 4)}},
            "architecture_small": {"func": self._evolve_architecture, "cost": 2.0, "params": {"scale_factor_range": (1.1, 1.3), "layer_increase_range": (1, 2)}},
            "parameters_global": {"func": self._evolve_learning_parameters, "cost": 1.0, "params": {"scope": "global"}},
            "parameters_targeted": {"func": self._evolve_learning_parameters, "cost": 1.5, "params": {"scope": "targeted"}},
            "paradigm_shift": {"func": self._evolve_learning_paradigm, "cost": 4.0},
            "neuron_type_trial": {"func": self._evolve_neuron_type, "cost": 3.0},
            "lr_rule_param_opt": {"func": self._evolve_learning_rule_params, "cost": 2.5},
            "apply_social_recipe": {"func": self._apply_social_evolution_recipe, "cost": 0.5},
        }

        if HSEO_AVAILABLE:
            self.evolution_operators["hseo_optimize_lp"] = {
                "func": self._hseo_optimize_learning_params,
                "cost": 6.0,
                "params": {
                    "param_keys": ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight", "training.gradient_based.loss.sparsity_reg_weight"],
                    "hseo_iterations": 10,
                    "hseo_particles": 5
                }
            }

        self.evolution_success_rates: Dict[str, Tuple[float, int]] = {
            op_name: (0.5, 0) for op_name in self.evolution_operators
        }

        # ハードウェアコンパイラ
        self.hardware_compiler = NeuromorphicCompiler()

        print(
            f"🧬 Master Self-Evolving Agent initialized. (HSEO Enabled: {HSEO_AVAILABLE})")

    async def evolve(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> str:
        """進化サイクルを実行"""
        logging.info("--- Initiating **Master** Evolution Cycle ---")
        logging.info(
            f"   - Meta-Cognition Status: {performance_eval.get('status', 'unknown')}")

        current_budget = self.evolution_budget
        applied_evolutions: List[str] = []
        final_message = "Evolution cycle completed. "

        # 1. 社会学習
        if random.random() < self.social_learning_probability:
            logging.info("   - Considering social learning...")
            social_op_name = "apply_social_recipe"
            if social_op_name in self.evolution_operators and current_budget >= self.evolution_operators[social_op_name]["cost"]:
                social_op = self.evolution_operators[social_op_name]
                social_result_msg = await social_op["func"](performance_eval, internal_state)

                success = "successfully" in social_result_msg.lower()
                self._update_evolution_history(social_op_name, success)
                applied_evolutions.append(
                    f"SocialRecipe ({'Success' if success else 'Fail'}): {social_result_msg}")
                if success:
                    current_budget -= social_op["cost"]

        # 2. 予算内での進化試行
        attempts = 0
        max_attempts = 3
        while current_budget > 0 and attempts < max_attempts:
            attempts += 1
            priorities = self._determine_evolution_priorities_v2(
                performance_eval, internal_state)
            chosen_op_name = self._select_evolution_operator(
                priorities, current_budget)

            if not chosen_op_name:
                break

            chosen_op = self.evolution_operators[chosen_op_name]
            evolution_func: Callable[..., Any] = chosen_op["func"]
            op_params: Dict[str, Any] = chosen_op.get("params", {})

            result_message: str
            try:
                if asyncio.iscoroutinefunction(evolution_func):
                    result_message = await evolution_func(performance_eval, internal_state, **op_params)
                else:
                    result_message = evolution_func(
                        performance_eval, internal_state, **op_params)

                success = "failed" not in result_message.lower()
            except Exception as e:
                logging.error(
                    f"Error executing evolution operator '{chosen_op_name}': {e}")
                result_message = f"Failed with error: {e}"
                success = False

            self._update_evolution_history(chosen_op_name, success)
            current_budget -= chosen_op["cost"]
            applied_evolutions.append(
                f"{chosen_op_name} ({'Success' if success else 'Fail'})")

            if success and "New config" in result_message:
                import re
                match = re.search(r"New config: '([^']+)'", result_message)
                if match:
                    new_path = match.group(1)
                    if "model" in chosen_op_name or "architecture" in chosen_op_name:
                        self.model_config_path = new_path
                    else:
                        self.training_config_path = new_path

            self.memory.record_experience(
                state=internal_state,
                action="self_evolution_step",
                result={"operator": chosen_op_name,
                        "message": result_message, "success": success},
                reward={"internal": 0.5 if success else -0.5}, expert_used=["self_evolver"],
                decision_context={"reason": "Attempting self-evolution."}
            )

        final_message += f"Applied: {', '.join(applied_evolutions)}. Remaining budget: {current_budget:.1f}"
        logging.info(final_message)
        return final_message

    def _determine_evolution_priorities_v2(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> Dict[str, float]:
        priorities: Dict[str, float] = {
            op_name: 1.0 for op_name in self.evolution_operators}
        status: Optional[str] = str(performance_eval.get("status"))

        if status == "capability_gap":
            priorities["architecture_large"] *= 2.5
            priorities["neuron_type_trial"] *= 1.8
        elif status == "knowledge_gap":
            priorities["paradigm_shift"] *= 2.0
            priorities["parameters_targeted"] *= 1.8

        if internal_state.get("boredom", 0.0) > 0.8:
            priorities["paradigm_shift"] *= 2.0

        return priorities

    def _select_evolution_operator(self, priorities: Dict[str, float], current_budget: float) -> Optional[str]:
        weighted_priorities: Dict[str, float] = {}
        candidate_operators = {
            op_name: op_data for op_name, op_data in self.evolution_operators.items()
            if op_data["cost"] <= current_budget
        }
        if not candidate_operators:
            return None

        for op_name in candidate_operators.keys():
            base_priority = priorities.get(op_name, 1.0)
            success_rate, trials = self.evolution_success_rates.get(
                op_name, (0.5, 0))
            confidence = 1.0 - math.exp(-trials / 10.0)
            weight = base_priority * \
                (confidence * success_rate + (1 - confidence) * 0.5)
            weighted_priorities[op_name] = max(0.01, weight)

        evolution_types = list(weighted_priorities.keys())
        probabilities = [weighted_priorities[op] for op in evolution_types]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return random.choices(evolution_types, weights=probabilities, k=1)[0]

    def _update_evolution_history(self, op_name: str, success: bool) -> None:
        current_rate, trials = self.evolution_success_rates.get(
            op_name, (0.5, 0))
        new_trials = trials + 1
        new_rate = (current_rate * trials +
                    (1.0 if success else 0.0)) / new_trials
        self.evolution_success_rates[op_name] = (new_rate, new_trials)

    def _save_evolved_config(self, config: Any, original_path: str, suffix: str = "evolved") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(original_path)
        name_without_ext, ext = os.path.splitext(base_name)
        new_filename = f"{name_without_ext}_{suffix}_{timestamp}_v{self.get_next_version()}{ext}"
        new_config_path = os.path.join(self.evolved_config_dir, new_filename)
        OmegaConf.save(config=config, f=new_config_path)
        return new_config_path

    # --- 進化オペレータ群 ---

    def _evolve_architecture(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                             scale_factor_range: Tuple[float, float] = (
                                 1.1, 1.3),
                             layer_increase_range: Tuple[int, int] = (1, 2),
                             **kwargs: Any) -> str:
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            return "Architecture evolution failed: model_config_path invalid."
        try:
            logging.info(
                f"🧬 Evolving architecture ({self.model_config_path})...")
            cfg = OmegaConf.load(self.model_config_path)
            scale_factor = random.uniform(*scale_factor_range)
            layer_increase = random.randint(*layer_increase_range)

            cfg.model.d_model = int(cfg.model.get(
                "d_model", 128) * scale_factor)
            cfg.model.num_layers = cfg.model.get(
                "num_layers", 4) + layer_increase

            try:
                temp_model = SNNCore(config=cfg.model, vocab_size=1000)
                hw_stats = self.hardware_compiler.compile(temp_model)
                logging.info(
                    f"   - ⚡️ Hardware Estimation: {hw_stats.get('estimated_power_mW', 0.0):.2f} mW")
            except Exception as hw_e:
                logging.warning(f"   - ⚠️ Hardware check failed: {hw_e}")

            new_config_path = self._save_evolved_config(
                cfg, self.model_config_path, suffix="arch")
            return f"Successfully evolved architecture. New config: '{new_config_path}'."
        except Exception as e:
            return f"Architecture evolution failed: {e}"

    def _evolve_learning_parameters(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                                    scope: str = "global", **kwargs: Any) -> str:
        if not self.training_config_path or not os.path.exists(self.training_config_path):
            return "Parameter evolution failed: training_config_path invalid."

        try:
            logging.info(f"🧬 Evolving learning parameters (Scope: {scope})...")
            cfg = OmegaConf.load(self.training_config_path)

            # 変異幅の設定
            mutation_strength = 0.2 if scope == "global" else 0.5

            # Learning Rate Mutation
            current_lr = cfg.training.gradient_based.get("learning_rate", 1e-3)
            new_lr = current_lr * \
                random.uniform(1.0 - mutation_strength,
                               1.0 + mutation_strength)
            cfg.training.gradient_based.learning_rate = new_lr

            # Weight Decay Mutation
            current_wd = cfg.training.gradient_based.loss.get(
                "weight_decay", 1e-4)
            new_wd = current_wd * \
                random.uniform(1.0 - mutation_strength,
                               1.0 + mutation_strength)
            cfg.training.gradient_based.loss.weight_decay = new_wd

            new_config_path = self._save_evolved_config(
                cfg, self.training_config_path, suffix="params")
            return f"Successfully evolved parameters (lr={new_lr:.2e}, wd={new_wd:.2e}). New config: '{new_config_path}'."

        except Exception as e:
            return f"Parameter evolution failed: {e}"

    def _evolve_learning_paradigm(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        # Placeholder for future paradigm shift logic (e.g. switching from Backprop to STDP)
        return "Paradigm shift not yet implemented."

    def _evolve_neuron_type(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            return "Neuron type evolution failed: model_config_path invalid."

        try:
            logging.info("🧬 Evolving neuron type...")
            cfg = OmegaConf.load(self.model_config_path)

            candidate_types = ["LIF", "PLIF", "Izhikevich"]
            current_type = cfg.model.neuron.get("type", "LIF")

            # 現在と異なるタイプを抽選
            candidates = [t for t in candidate_types if t != current_type]
            if not candidates:
                return "No alternative neuron types available."

            new_type = random.choice(candidates)
            cfg.model.neuron.type = new_type

            # タイプ変更に伴うパラメータ調整（必要であれば）
            if new_type == "Izhikevich":
                cfg.model.neuron.a = 0.02
                cfg.model.neuron.b = 0.2

            new_config_path = self._save_evolved_config(
                cfg, self.model_config_path, suffix="neuron")
            return f"Successfully evolved neuron type to '{new_type}'. New config: '{new_config_path}'."

        except Exception as e:
            return f"Neuron evolution failed: {e}"

    def _evolve_learning_rule_params(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        # Placeholder for STDP rule parameter evolution
        return "Learning rule param evolution not yet implemented."

    async def _apply_social_evolution_recipe(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        return "Applied social recipe (implied)"

    def _hseo_optimize_learning_params(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        return "HSEO optimized (implied)"

    def get_next_version(self) -> int:
        return random.randint(1000, 9999)
