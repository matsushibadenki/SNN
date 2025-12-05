# ファイルパス: snn_research/agent/self_evolving_agent.py
# Title: Master Self-Evolving Agent (Async Fix)
# Description: 
#   メタ認知に基づく自己診断、HSEOを用いたハイパーパラメータ最適化、
#   アーキテクチャ探索、および社会学習を統合したマスター進化エージェント。
#   修正: 非同期処理のネスト問題（Running Loop内でのasyncio.run）を解決するため、
#   evolveメソッド全体を非同期化し、awaitで制御するように変更。

from typing import Dict, Any, Optional, List, Tuple, cast, Callable, Union
import os
import random
import math
import asyncio
import logging
import numpy as np
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

# HSEO最適化モジュール
HSEO_AVAILABLE = False
try:
    from snn_research.optimization.hseo import optimize_with_hseo, evaluate_snn_params
    HSEO_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ HSEO module not found. 'hseo_optimize_lp' operator will be disabled.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SelfEvolvingAgentMaster(AutonomousAgent):
    """
    Phase 5: ニューロシンボリック進化を司るマスターエージェント。
    メタ認知SNNからの診断に基づき、自身の脳（モデル構造・学習則）を書き換える権限を持つ。
    """
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
        super().__init__(name, planner, model_registry, memory, web_crawler)
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
        
        self.evolved_config_dir = os.path.join("runs", "evolved_configs")
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
        
        # HSEOが利用可能なら高コスト・高精度の最適化オペレータを追加
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
        
        # ハードウェアコンパイラ（進化後の効率性チェック用）
        self.hardware_compiler = NeuromorphicCompiler(hardware_profile_name="loihi")
        
        print(f"🧬 Master Self-Evolving Agent initialized. (HSEO Enabled: {HSEO_AVAILABLE})")

    async def evolve(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> str:
        """
        進化サイクルを実行するメインメソッド（非同期版）。
        メタ認知評価に基づいて最適な進化戦略を選択し、実行する。
        """
        logging.info("--- Initiating **Master** Evolution Cycle ---")
        logging.info(f"   - Meta-Cognition Status: {performance_eval.get('status', 'unknown')}")
        logging.info(f"   - Internal State: Curiosity={internal_state.get('curiosity', 0.0):.2f}, Boredom={internal_state.get('boredom', 0.0):.2f}")
        
        current_budget = self.evolution_budget
        applied_evolutions: List[str] = []
        final_message = "Evolution cycle completed. "

        # 1. 社会学習（他エージェントの成功事例の模倣）の検討
        if random.random() < self.social_learning_probability:
             logging.info("   - Considering social learning...")
             social_op_name = "apply_social_recipe"
             if social_op_name in self.evolution_operators and current_budget >= self.evolution_operators[social_op_name]["cost"]:
                 social_op = self.evolution_operators[social_op_name]
                 # 非同期実行 (await)
                 social_result_msg = await social_op["func"](performance_eval, internal_state)
                 
                 success = "successfully" in social_result_msg.lower()
                 self._update_evolution_history(social_op_name, success)
                 applied_evolutions.append(f"SocialRecipe ({'Success' if success else 'Fail'}): {social_result_msg}")
                 
                 if success:
                     current_budget -= social_op["cost"]
                     final_message += "Applied successful evolution recipe from another agent. "

        # 2. 予算内での進化試行
        attempts = 0
        max_attempts = 3
        while current_budget > 0 and attempts < max_attempts:
            attempts += 1
            logging.info(f"\n   --- Evolution Attempt {attempts} (Budget: {current_budget:.1f}) ---")
            
            # 優先順位の決定
            priorities = self._determine_evolution_priorities_v2(performance_eval, internal_state)
            chosen_op_name = self._select_evolution_operator(priorities, current_budget)

            if not chosen_op_name:
                logging.info("   - No suitable evolution operator found within budget or based on priorities.")
                break

            chosen_op = self.evolution_operators[chosen_op_name]
            logging.info(f"   - Chosen Operator: {chosen_op_name} (Cost: {chosen_op['cost']})")

            evolution_func: Callable[..., Any] = chosen_op["func"]
            op_params: Dict[str, Any] = chosen_op.get("params", {})

            # 実行
            result_message: str
            try:
                # 関数がコルーチンかどうかを判定して呼び出し分ける
                if asyncio.iscoroutinefunction(evolution_func):
                    result_message = await evolution_func(performance_eval, internal_state, **op_params)
                else:
                    # 同期関数の場合はそのまま実行（ブロックするが、計算負荷の高い処理は別プロセスかスレッドが望ましい）
                    # ここでは簡易的にメインスレッドで実行
                    result_message = evolution_func(performance_eval, internal_state, **op_params)
                
                success = "failed" not in result_message.lower()
            except Exception as e:
                logging.error(f"Error executing evolution operator '{chosen_op_name}': {e}")
                result_message = f"Failed with error: {e}"
                success = False

            # 結果の記録と履歴更新
            self._update_evolution_history(chosen_op_name, success)
            current_budget -= chosen_op["cost"]
            applied_evolutions.append(f"{chosen_op_name} ({'Success' if success else 'Fail'})")

            # 成功時にパスを更新
            if success and "New config" in result_message:
                import re
                match = re.search(r"New config: '([^']+)'", result_message)
                if match:
                    new_path = match.group(1)
                    if "model" in chosen_op_name or "architecture" in chosen_op_name or "neuron" in chosen_op_name:
                        self.model_config_path = new_path
                        logging.info(f"   - Model config path updated to: {new_path}")
                    else:
                        self.training_config_path = new_path
                        logging.info(f"   - Training config path updated to: {new_path}")

            # メモリへの記録
            self.memory.record_experience(
                state=self.current_state, action="self_evolution_step",
                result={"operator": chosen_op_name, "message": result_message, "success": success, "budget_spent": chosen_op["cost"]},
                reward={"internal": 0.5 if success else -0.5}, expert_used=["self_evolver"],
                decision_context={"reason": "Attempting self-evolution.", "performance_eval": performance_eval, "internal_state": internal_state, "chosen_operator": chosen_op_name}
            )

        final_message += f"Applied evolutions: {', '.join(applied_evolutions)}. Remaining budget: {current_budget:.1f}"
        logging.info(final_message)
        return final_message

    def _determine_evolution_priorities_v2(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> Dict[str, float]:
        """
        状態に応じて進化オペレータの優先順位を動的に変更するロジック。
        """
        priorities: Dict[str, float] = {op_name: 1.0 for op_name in self.evolution_operators}
        status: Optional[str] = str(performance_eval.get("status"))

        # ステータス別戦略
        if status == "capability_gap":
            # 能力不足 -> アーキテクチャ拡張やニューロンタイプ変更を優先
            priorities["architecture_large"] *= 2.5
            priorities["architecture_small"] *= 1.5
            priorities["neuron_type_trial"] *= 1.8
            priorities["paradigm_shift"] *= 1.2
        elif status == "knowledge_gap":
            # 知識不足 -> 学習パラメータ調整やパラダイムシフトを優先
            priorities["paradigm_shift"] *= 2.0
            priorities["parameters_targeted"] *= 1.8
            priorities["lr_rule_param_opt"] *= 1.5
            priorities["parameters_global"] *= 1.2
        elif status == "learning":
             # 学習中 -> パラメータ微調整
             priorities["parameters_global"] *= 1.5
             priorities["lr_rule_param_opt"] *= 1.2

        # 内部状態による修飾
        if internal_state.get("boredom", 0.0) > 0.8:
            # 退屈 -> 大胆な変更や社会学習
            priorities["paradigm_shift"] *= 2.0
            priorities["architecture_large"] *= 1.5
            priorities["neuron_type_trial"] *= 1.3
            priorities["apply_social_recipe"] *= 1.5
            
        if internal_state.get("curiosity", 0.0) > 0.8:
            context: Any = internal_state.get("curiosity_context")
            if isinstance(context, str):
                if "efficiency" in context or "energy" in context:
                    priorities["architecture_small"] *= 1.5
                    priorities["parameters_targeted"] *= 1.4
                elif "accuracy" in context or "performance" in context:
                    priorities["architecture_large"] *= 1.5
            
        # HSEOの優先度調整
        if "hseo_optimize_lp" in priorities:
             if status in ["knowledge_gap", "learning"]:
                 priorities["hseo_optimize_lp"] *= 1.5
             elif internal_state.get("curiosity", 0.0) > 0.9:
                 priorities["hseo_optimize_lp"] *= 1.3
             else:
                 priorities["hseo_optimize_lp"] *= 0.8 # コストが高いため通常は抑制

        return priorities

    def _select_evolution_operator(self, priorities: Dict[str, float], current_budget: float) -> Optional[str]:
        """予算と優先度、過去の成功率に基づいてオペレータを選択する"""
        weighted_priorities: Dict[str, float] = {}
        total_weight = 0.0
        candidate_operators = {
            op_name: op_data for op_name, op_data in self.evolution_operators.items()
            if op_data["cost"] <= current_budget
        }
        if not candidate_operators:
            return None

        for op_name in candidate_operators.keys():
            base_priority = priorities.get(op_name, 1.0)
            # 過去の成功体験を加味 (Confidence based)
            success_rate, trials = self.evolution_success_rates.get(op_name, (0.5, 0))
            confidence = 1.0 - math.exp(-trials / 10.0)
            weight = base_priority * (confidence * success_rate + (1 - confidence) * 0.5)
            
            weighted_priorities[op_name] = max(0.01, weight)
            total_weight += weighted_priorities[op_name]

        if total_weight == 0:
            return random.choice(list(candidate_operators.keys()))

        evolution_types: List[str] = list(weighted_priorities.keys())
        probabilities: List[float] = [weighted_priorities[op] / total_weight for op in evolution_types]
        
        chosen_type = random.choices(evolution_types, weights=probabilities, k=1)[0]
        return chosen_type

    def _update_evolution_history(self, op_name: str, success: bool) -> None:
        current_rate, trials = self.evolution_success_rates.get(op_name, (0.5, 0))
        new_trials = trials + 1
        new_rate = (current_rate * trials + (1.0 if success else 0.0)) / new_trials
        self.evolution_success_rates[op_name] = (new_rate, new_trials)
        logging.info(f"   - Updated '{op_name}' success rate: {new_rate:.2f} (after {new_trials} trials)")

    def _save_evolved_config(self, config: Any, original_path: str, suffix: str = "evolved") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(original_path)
        name_without_ext, ext = os.path.splitext(base_name)
        new_filename = f"{name_without_ext}_{suffix}_{timestamp}_v{self.get_next_version()}{ext}"
        new_config_path = os.path.join(self.evolved_config_dir, new_filename)
        try:
            OmegaConf.save(config=config, f=new_config_path)
            logging.info(f"   - Saved evolved config to: {new_config_path}")
            return new_config_path
        except Exception as e:
            logging.error(f"Failed to save evolved config: {e}")
            fallback_path = f"{original_path}.tmp.{timestamp}"
            OmegaConf.save(config=config, f=fallback_path)
            return fallback_path

    # --- 進化オペレータ群 ---

    def _evolve_architecture(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                             scale_factor_range: Tuple[float, float] = (1.1, 1.3),
                             layer_increase_range: Tuple[int, int] = (1, 2),
                             **kwargs: Any) -> str:
        """モデルアーキテクチャ（層の深さ、幅）を進化させる"""
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            return "Architecture evolution failed: model_config_path is not set or file not found."
        try:
            logging.info(f"🧬 Evolving architecture ({self.model_config_path}) with scale {scale_factor_range}, layers {layer_increase_range}...")
            cfg = OmegaConf.load(self.model_config_path)
            scale_factor = random.uniform(*scale_factor_range)
            layer_increase = random.randint(*layer_increase_range)

            original_d_model = cfg.model.get("d_model", 128)
            original_num_layers = cfg.model.get("num_layers", 4)

            cfg.model.d_model = int(original_d_model * scale_factor)
            cfg.model.num_layers = original_num_layers + layer_increase

            if 'd_state' in cfg.model: 
                cfg.model.d_state = int(cfg.model.get('d_state', 64) * scale_factor)
            
            if 'n_head' in cfg.model:
                 current_n_head = cfg.model.get('n_head', 2)
                 # d_modelが増えたらヘッド数も増やす
                 if scale_factor > 1.4 and cfg.model.d_model % (current_n_head * 2) == 0:
                      cfg.model.n_head = current_n_head * 2
                      logging.info(f"   - n_head increased: {current_n_head} -> {cfg.model.n_head}")
            
            # Neuron branch調整
            if 'neuron' in cfg.model and 'branch_features' in cfg.model.neuron:
                 num_branches = cfg.model.neuron.get("num_branches", 4)
                 required_d_model = (cfg.model.d_model // num_branches) * num_branches
                 if required_d_model != cfg.model.d_model:
                      logging.info(f"   - Adjusting d_model: {cfg.model.d_model} -> {required_d_model}")
                      cfg.model.d_model = required_d_model
                 if cfg.model.d_model > 0:
                     cfg.model.neuron.branch_features = max(16, cfg.model.d_model // num_branches)

            logging.info(f"   - d_model: {original_d_model} -> {cfg.model.d_model}, num_layers: {original_num_layers} -> {cfg.model.num_layers}")
            
            # ハードウェア適合性チェック (コンパイラシミュレーション)
            try:
                temp_model = SNNCore(config=cfg.model, vocab_size=1000)
                temp_compile_path = os.path.join(self.evolved_config_dir, f"hw_config_temp_{random.randint(0,9999)}.yaml")
                self.hardware_compiler.compile(temp_model, temp_compile_path)
                
                # 推定値を計算
                total_spikes_est = 1000 * 16
                hw_report = self.hardware_compiler.simulate_on_hardware(temp_compile_path, total_spikes_est, time_steps=16)
                logging.info(f"   - ⚡️ Hardware Estimation: {hw_report}")
                
                if os.path.exists(temp_compile_path):
                    os.remove(temp_compile_path)

            except Exception as hw_e:
                logging.warning(f"   - ⚠️ Hardware compilation check failed (ignoring for evolution): {hw_e}")

            new_config_path = self._save_evolved_config(cfg, self.model_config_path, suffix="arch")
            return f"Successfully evolved architecture. New config: '{new_config_path}'."
        except Exception as e: 
            return f"Architecture evolution failed: {e}"

    def _evolve_learning_parameters(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                                    scope: str = "global", **kwargs: Any) -> str:
        """学習率や正則化パラメータなどのハイパーパラメータを進化させる"""
        if not self.training_config_path or not os.path.exists(self.training_config_path): 
            return "LP evo failed: training_config_path not found."
        try:
            logging.info(f"🧠 Evolving LP ({self.training_config_path}), scope: {scope}...")
            cfg = OmegaConf.load(self.training_config_path)
            params_to_evolve: List[str] = []
            
            if scope == "global": 
                params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight", "training.gradient_based.loss.sparsity_reg_weight", "training.gradient_based.loss.temporal_compression_weight"]
            elif scope == "targeted":
                if performance_eval.get("status") == "knowledge_gap": 
                    params_to_evolve = ["training.gradient_based.learning_rate", "training.biologically_plausible.learning_rule", "training.biologically_plausible.neuron.tau_mem", "training.biologically_plausible.neuron.base_threshold"]
                elif internal_state.get("curiosity", 0.0) > 0.8: 
                    params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.distillation.temperature"]
                else: 
                    params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight"]
            else: 
                return f"Unknown scope: {scope}"

            valid_params = [p for p in params_to_evolve if OmegaConf.select(cfg, p, default=None) is not None]
            if not valid_params: 
                return f"No valid params for scope '{scope}'."
            
            param_key = random.choice(valid_params)
            original_value = OmegaConf.select(cfg, param_key)

            change_factor = random.uniform(0.7, 1.3)
            new_value: Any
            
            if internal_state.get("curiosity", 0.0) > 0.8 and scope == "targeted": 
                change_factor = random.uniform(0.5, 1.5)

            if isinstance(original_value, float): 
                new_value = max(1e-7, original_value * change_factor)
            elif isinstance(original_value, int): 
                new_value = max(1, int(original_value * change_factor))
            elif isinstance(original_value, str) and "learning_rule" in param_key:
                 rules = ["STDP", "REWARD_MODULATED_STDP", "CAUSAL_TRACE_V2", "PROBABILISTIC_HEBBIAN"]
                 candidates = [r for r in rules if r != original_value]
                 new_value = random.choice(candidates) if candidates else original_value
            else: 
                logging.warning(f"   - Param '{param_key}' type ({type(original_value)}) not handled.")
                return f"Skipped '{param_key}'."

            OmegaConf.update(cfg, param_key, new_value, merge=True)
            logging.info(f"   - Evolved '{param_key}': {original_value} -> {new_value}")
            
            new_config_path = self._save_evolved_config(cfg, self.training_config_path, suffix="lp")
            return f"Successfully evolved LP (scope: {scope}). New config: '{new_config_path}'."
        except Exception as e: 
            return f"LP evo (scope: {scope}) failed: {e}"

    def _evolve_learning_paradigm(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        """学習パラダイム（例：勾配法 -> 生物学的学習）を切り替える"""
        if not self.training_config_path or not os.path.exists(self.training_config_path): 
            return "Paradigm evo failed: training_config_path not found."
        try:
            logging.info(f"🔄 Evolving paradigm ({self.training_config_path})...")
            cfg = OmegaConf.load(self.training_config_path)
            current_paradigm = cfg.training.get("paradigm", "gradient_based")
            available = [ "gradient_based", "self_supervised", "physics_informed", "bio-causal-sparse", "bio-particle-filter", "bio-probabilistic-hebbian" ]
            candidates = [p for p in available if p != current_paradigm]
            
            if not candidates: return "No alternatives."

            chosen_paradigm: str
            status = performance_eval.get("status")
            
            if status == "knowledge_gap": 
                priority = [p for p in candidates if p.startswith("bio-") or p == "self_supervised"]
                chosen_paradigm = random.choice(priority) if priority else random.choice(candidates)
            elif internal_state.get("boredom", 0.0) > 0.85: 
                chosen_paradigm = random.choice(candidates)
            elif status == "capability_gap": 
                priority = ["gradient_based", "bio-probabilistic-hebbian", "bio-particle-filter"]
                valid = [p for p in priority if p in candidates]
                chosen_paradigm = random.choice(valid) if valid else random.choice(candidates)
            else: 
                chosen_paradigm = random.choice(candidates)

            cfg.training.paradigm = chosen_paradigm
            logging.info(f"   - Paradigm evolved: '{current_paradigm}' -> '{chosen_paradigm}'")
            
            new_config_path = self._save_evolved_config(cfg, self.training_config_path, suffix="paradigm")
            return f"Successfully evolved paradigm to '{chosen_paradigm}'. New config: '{new_config_path}'."
        except Exception as e: 
            return f"Paradigm evo failed: {e}"

    def _evolve_neuron_type(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        """ニューロンモデル（LIF, Izhikevichなど）を切り替える"""
        if not self.model_config_path or not os.path.exists(self.model_config_path): 
            return "Neuron evo failed: model_config_path not found."
        try:
            logging.info(f"💡 Evolving neuron type in {self.model_config_path}...")
            cfg = OmegaConf.load(self.model_config_path)
            current = cfg.model.neuron.get("type", "lif")
            available = ["lif", "izhikevich"]
            candidates = [nt for nt in available if nt != current]
            
            if not candidates: 
                return f"No alternatives (current: {current})."
            
            new = random.choice(candidates)
            cfg.model.neuron.type = new
            logging.info(f"   - Neuron type evolved: '{current}' -> '{new}'")
            
            if new == "izhikevich" and "a" not in cfg.model.neuron: 
                cfg.model.neuron.a=0.02
                cfg.model.neuron.b=0.2
                cfg.model.neuron.c=-65.0
                cfg.model.neuron.d=8.0
                logging.info("   - Added default Izhikevich params.")
            
            new_config_path = self._save_evolved_config(cfg, self.model_config_path, suffix="neuron")
            return f"Successfully evolved neuron type to '{new}'. New config: '{new_config_path}'."
        except Exception as e: 
            return f"Neuron evo failed: {e}"

    def _evolve_learning_rule_params(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        """生物学的学習則のパラメータを進化させる"""
        if not self.training_config_path or not os.path.exists(self.training_config_path): 
            return "LR param evo failed: training_config_path not found."
        try:
            logging.info(f"⚙️ Evolving LR params in {self.training_config_path}...")
            cfg = OmegaConf.load(self.training_config_path)
            current_rule = cfg.training.biologically_plausible.get("learning_rule", "CAUSAL_TRACE_V2")
            rule_key: Optional[str] = None
            params_list: List[str] = []
            
            if current_rule == "STDP": 
                rule_key = "stdp"
                params_list = ["lr", "a+", "a-", "tau_t"]
            elif current_rule == "REWARD_MODULATED_STDP": 
                rule_key = "reward_modulated_stdp"
                params_list = ["lr", "tau_e"]
            elif current_rule.startswith("CAUSAL_TRACE"): 
                rule_key = "causal_trace"
                params_list = ["lr", "tau_e", "cdt", "dlrf", "cms", "ckr", "rlrf"]
            elif current_rule == "PROBABILISTIC_HEBBIAN": 
                rule_key = "probabilistic_hebbian"
                params_list = ["lr", "wd"]
            else: 
                return f"Rule '{current_rule}' not supported."

            full_key_base = f"training.biologically_plausible.{rule_key}"
            if OmegaConf.select(cfg, full_key_base, default=None) is None: 
                return f"Config section '{full_key_base}' not found."
                
            abbr_map = {"learning_rate": "lr", "a_plus": "a+", "a_minus": "a-", "tau_trace": "tau_t", "tau_eligibility": "tau_e", "credit_time_decay": "cdt", "dynamic_lr_factor": "dlrf", "context_modulation_strength": "cms", "competition_k_ratio": "ckr", "rule_based_lr_factor": "rlrf", "weight_decay": "wd"}
            
            full_params_list = [name for abbr, name in abbr_map.items() if abbr in params_list]
            valid_params = [p for p in full_params_list if OmegaConf.select(cfg, f"{full_key_base}.{p}", default=None) is not None]
            
            if not valid_params: 
                return f"No valid params for '{current_rule}'."
            
            param_name = random.choice(valid_params)
            param_key = f"{full_key_base}.{param_name}"

            original_value = OmegaConf.select(cfg, param_key)
            if not isinstance(original_value, (float, int)): 
                return f"Param '{param_name}' not numeric."
            
            change = random.uniform(0.8, 1.2)
            new_value = max(1e-7, original_value * change) if isinstance(original_value, float) else max(1, int(original_value * change))

            OmegaConf.update(cfg, param_key, new_value, merge=True)
            logging.info(f"   - Evolved '{param_key}': {original_value} -> {new_value}")
            
            new_config_path = self._save_evolved_config(cfg, self.training_config_path, suffix="lr_params")
            return f"Successfully evolved LR params. New config: '{new_config_path}'."
        except Exception as e: 
            return f"LR param evo failed: {e}"

    async def _apply_social_evolution_recipe(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        """モデルレジストリを参照し、優れた他モデルの設定を模倣する（社会学習）"""
        logging.info("🤝 Attempting social learning via Model Registry imitation...")
        
        try:
            all_models = await self.model_registry.list_models()
            if not all_models:
                return "Social learning failed: No models in registry."
            
            sorted_models = sorted(
                all_models, 
                key=lambda x: x.get("metrics", {}).get("accuracy", 0.0), 
                reverse=True
            )
            
            mentor_model = None
            for m in sorted_models[:3]:
                if m.get("config"):
                    mentor_model = m
                    break
            
            if not mentor_model:
                return "Social learning failed: No suitable mentor model found with config."
                
            mentor_config = mentor_model.get("config", {})
            mentor_id = mentor_model.get("model_id", "unknown")
            logging.info(f"   - Selected mentor: {mentor_id} (Acc: {mentor_model.get('metrics', {}).get('accuracy', 0.0):.3f})")

            mentor_lr = None
            try:
                if "training" in mentor_config:
                    mentor_lr = mentor_config["training"]["gradient_based"]["learning_rate"]
                elif "learning_rate" in mentor_config:
                    mentor_lr = mentor_config["learning_rate"]
            except (KeyError, TypeError):
                pass
            
            if mentor_lr:
                if not self.training_config_path or not os.path.exists(self.training_config_path):
                    return "Social learning failed: Training config path invalid."
                
                cfg = OmegaConf.load(self.training_config_path)
                current_lr = OmegaConf.select(cfg, "training.gradient_based.learning_rate")
                
                OmegaConf.update(cfg, "training.gradient_based.learning_rate", mentor_lr, merge=True)
                logging.info(f"   - Mimicked Learning Rate: {current_lr} -> {mentor_lr}")
                
                new_config_path = self._save_evolved_config(cfg, self.training_config_path, suffix="social")
                
                self.training_config_path = new_config_path
                return f"Successfully applied social recipe from {mentor_id}. New config: '{new_config_path}'."
            else:
                return f"Social learning failed: Could not extract learning parameters from mentor {mentor_id}."

        except Exception as e:
            return f"Applying social recipe failed: {e}"

    def _hseo_optimize_learning_params(
        self,
        performance_eval: Dict[str, Any],
        internal_state: Dict[str, Any],
        param_keys: List[str] = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight"],
        hseo_iterations: int = 10,
        hseo_particles: int = 5,
        **kwargs: Any
    ) -> str:
        """HSEO (Hybrid Swarm Evolution Optimization) を用いたハイパーパラメータ最適化"""
        if not HSEO_AVAILABLE:
            return "HSEO optimization skipped: HSEO module not available."
            
        if not self.training_config_path or not os.path.exists(self.training_config_path):
            return "HSEO LP evo failed: training_config_path not found."
        if not self.model_config_path:
             return "HSEO LP evo failed: model_config_path not set."

        try:
            logging.info(f"⚙️ Optimizing LP using HSEO ({self.training_config_path})...")
            cfg = OmegaConf.load(self.training_config_path)

            initial_params: List[float] = []
            param_bounds: List[Tuple[float, float]] = []
            valid_param_keys: List[str] = []
            
            for key in param_keys:
                value = OmegaConf.select(cfg, key, default=None)
                if value is None or not isinstance(value, (float, int)):
                    logging.warning(f"   - Skipping non-numeric or missing param for HSEO: {key}")
                    continue
                
                float_value = float(value)
                initial_params.append(float_value)
                
                lower_bound: float
                upper_bound: float
                if "learning_rate" in key:
                    lower_bound = max(1e-7, float_value / 10.0)
                    upper_bound = min(1e-2, float_value * 10.0)
                elif "weight" in key:
                    lower_bound = max(0.0, float_value / 10.0)
                    upper_bound = min(1.0, float_value * 10.0 if float_value > 0 else 0.1)
                else:
                    lower_bound = float_value / 5.0
                    upper_bound = float_value * 5.0
                
                if lower_bound > upper_bound:
                    lower_bound, upper_bound = upper_bound, lower_bound
                
                param_bounds.append((lower_bound, upper_bound))
                valid_param_keys.append(key)

            if not valid_param_keys:
                return "HSEO LP evo failed: No valid parameters found to optimize."

            logging.info(f"   - Optimizing parameters: {valid_param_keys}")

            def objective_function(params_array: np.ndarray) -> np.ndarray:
                scores = np.zeros(params_array.shape[0])
                for i in range(params_array.shape[0]):
                    current_params: np.ndarray = params_array[i]
                    param_dict: Dict[str, Any] = {key: val for key, val in zip(valid_param_keys, current_params)}
                    
                    score: float = evaluate_snn_params(
                        model_config_path=cast(str, self.model_config_path),
                        base_training_config_path=cast(str, self.training_config_path),
                        params_to_override=param_dict,
                        eval_epochs=1, 
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        metric_to_optimize="loss"
                    )
                    scores[i] = score
                return scores
            
            best_params_np, best_score = optimize_with_hseo(
                objective_function=objective_function,
                dim=len(valid_param_keys),
                num_particles=hseo_particles,
                max_iterations=hseo_iterations,
                exploration_range=param_bounds,
                seed=random.randint(0, 10000),
                verbose=True
            )
            best_params: List[float] = best_params_np.tolist()
    
            logging.info(f"   - HSEO finished. Best score (loss): {best_score:.4f}")
    
            cfg_updated = OmegaConf.load(self.training_config_path)
            updated_keys: List[str] = []
            for key, value in zip(valid_param_keys, best_params):
                OmegaConf.update(cfg_updated, key, value, merge=True)
                updated_keys.append(key)
    
            logging.info(f"   - Updated parameters in config: {updated_keys}")
            
            new_config_path = self._save_evolved_config(cfg_updated, self.training_config_path, suffix="hseo")
    
            return f"Successfully optimized LP using HSEO. Best loss: {best_score:.4f}. New config: '{new_config_path}'."
    
        except Exception as e:
            import traceback
            logging.error(f"HSEO LP evo failed: {e}\n{traceback.format_exc()}")
            return f"HSEO LP evo failed: {e}"

    def get_next_version(self) -> int:
        return random.randint(1000, 9999)
