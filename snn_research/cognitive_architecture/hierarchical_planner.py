# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# (修正: pad_token 設定の追加)
# Title: 階層的プランナー (SNN-based Plan Generation)
# Description:
#   学習済み PlannerSNN モデルを使用して、高レベルな目標を具体的なタスクシーケンスに分解する。
#   ダミーのルールベースロジックを廃止し、モデルの推論結果(Logits)をデコードする。
#   修正: Tokenizerにpad_tokenが設定されていない場合のエラーを回避するため、eos_tokenをpad_tokenとして設定する処理を追加。

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import asyncio
import logging

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem

if TYPE_CHECKING:
    from snn_research.agent.memory import Memory

logger = logging.getLogger(__name__)

class Plan:
    def __init__(self, goal: str, task_list: List[Dict[str, Any]]):
        self.goal = goal
        self.task_list = task_list
    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', tasks={len(self.task_list)})"

class HierarchicalPlanner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        rag_system: RAGSystem,
        memory: "Memory",
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.rag_system = rag_system
        self.memory = memory
        self.planner_model = planner_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # --- ▼ 修正: pad_token の設定を追加 ▼ ---
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"HierarchicalPlanner: pad_token was None, set to eos_token ({self.tokenizer.eos_token})")
        # --- ▲ 修正 ▲ ---
        
        self.device = device
        if self.planner_model: 
            self.planner_model.to(self.device)
            self.planner_model.eval() # 推論モード
            
        # スキルマップの初期構築
        self.SKILL_MAP: Dict[int, Dict[str, Any]] = {} 
        
    async def _build_skill_map(self) -> Dict[int, Dict[str, Any]]:
        """
        モデルレジストリとツールセットから、IDとスキルの対応表を動的に構築する。
        PlannerSNNの出力クラスIDに対応させる。
        """
        all_models = await self.model_registry.list_models()
        skill_map: Dict[int, Dict[str, Any]] = {}
        
        # 1. モデルスキル (登録されている専門家モデル)
        for i, model_info in enumerate(all_models):
            skill_map[i] = {
                "task": model_info.get("model_id"),
                "description": model_info.get("task_description"),
                "expert_id": model_info.get("model_id"),
                "type": "model"
            }
        
        base_idx = len(skill_map)
        
        # 2. 固定ツールスキル (Calculator, Search)
        # 実際にはPlannerSNNの出力次元数(num_skills)に合わせてマッピングする必要がある
        # ここでは簡易的に末尾に追加
        tools = [
            {"task": "calculator", "description": "Perform mathematical calculations.", "expert_id": "tool_calculator", "type": "tool"},
            {"task": "web_search", "description": "Search the web for information.", "expert_id": "tool_web_crawler", "type": "tool"},
            {"task": "summarize", "description": "Summarize text.", "expert_id": "tool_summarizer", "type": "tool"},
        ]
        
        for j, tool in enumerate(tools):
            skill_map[base_idx + j] = tool
            
        return skill_map

    async def create_plan(self, high_level_goal: str, context: Optional[str] = None, skills_to_avoid: Optional[List[str]] = None) -> Plan:
        """
        SNNモデルを使用して目標から計画（タスクリスト）を生成する。
        """
        self.SKILL_MAP = await self._build_skill_map()
        if skills_to_avoid is None: skills_to_avoid = []
        
        task_list: List[Dict[str, Any]] = []

        # --- PlannerSNNによる推論 ---
        if self.planner_model:
            logger.info(f"🧠 PlannerSNN inferring plan for: '{high_level_goal}'")
            
            # 入力プロンプトの作成
            prompt = f"Goal: {high_level_goal} Context: {context or ''}"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            
            with torch.no_grad():
                # モデル出力: (Batch, NumSkills)
                # 複数のステップを出力するモデルなら (Batch, SeqLen, NumSkills) だが、
                # 現在のPlannerSNNは単一ステップの選択を行う設計と仮定（または繰り返し呼ぶ）
                logits = self.planner_model(inputs['input_ids'])
                
                # 上位k個のスキルを候補として取得、またはシーケンシャルに生成
                probs = F.softmax(logits, dim=-1)
                top_k = torch.topk(probs, k=3, dim=-1)
                indices = top_k.indices[0].tolist() # 最も確度の高いスキルID
                
                for skill_idx in indices:
                    skill = self.SKILL_MAP.get(skill_idx)
                    if skill and skill.get('task') not in skills_to_avoid:
                        # 重複を避ける簡易ロジック
                        if skill not in task_list:
                            task_list.append(skill)
                            
            if not task_list:
                logger.warning("PlannerSNN returned no valid skills. Falling back to heuristic.")
        else:
            logger.warning("PlannerSNN not loaded. Using heuristic fallback.")

        # --- フォールバック: ヒューリスティック (モデルがない場合や失敗時) ---
        if not task_list:
            task_list = self._heuristic_fallback(high_level_goal)

        return Plan(goal=high_level_goal, task_list=task_list)

    def _heuristic_fallback(self, prompt: str) -> List[Dict[str, Any]]:
        """最低限のルールベースフォールバック (PlannerSNN未学習時用)"""
        task_list = []
        prompt_lower = prompt.lower()
        
        # ツールマッチング
        if "calc" in prompt_lower or "+" in prompt_lower:
             # マップから検索
             for s in self.SKILL_MAP.values():
                 if s['task'] == 'calculator': task_list.append(s); break
        if "search" in prompt_lower or "find" in prompt_lower:
             for s in self.SKILL_MAP.values():
                 if s['task'] == 'web_search': task_list.append(s); break
                 
        # モデル検索
        if not task_list:
             for s in self.SKILL_MAP.values():
                 if s['type'] == 'model' and s.get('description') and s['description'] in prompt_lower:
                     task_list.append(s)
        
        return task_list

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        plan = asyncio.run(self.create_plan(task_request, context))
        return f"Plan created: {plan}"