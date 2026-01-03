# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# 日本語タイトル: Hierarchical Planner v2.2 (Type Safe & Emergent Compatible)
# 目的・内容:
#   高レベルの目標（Goal）を受け取り、PlannerSNNモデルを使用して
#   実行可能なサブタスクやスキル（Action Sequence）に分解する。
#   EmergentCognitiveSystemとの互換性レイヤーを追加。

import torch
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerBase

from snn_research.cognitive_architecture.planner_snn import PlannerSNN

logger = logging.getLogger(__name__)

@dataclass
class Plan:
    """EmergentCognitiveSystem用の計画コンテナ"""
    goal: str
    task_list: List[Dict[str, Any]] = field(default_factory=list)

class HierarchicalPlanner:
    """
    階層的プランナー:
    Goal -> (PlannerSNN) -> Skill/Action Selection -> Plan
    """
    def __init__(
        self,
        planner_model: PlannerSNN,
        tokenizer: PreTrainedTokenizerBase,
        action_space: Dict[int, str], # ID -> Action Name
        device: str = "cpu"
    ):
        self.model = planner_model
        self.tokenizer = tokenizer
        self.action_space = action_space
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"🗺️ Hierarchical Planner initialized with {len(action_space)} skills.")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        AsyncBrainKernelからの入力（Goal Text）を処理し、計画を生成する（同期実行用）。
        """
        goal_text = ""
        if isinstance(input_data, str):
            goal_text = input_data
        elif isinstance(input_data, dict) and "payload" in input_data:
            goal_text = str(input_data["payload"])
        else:
            goal_text = str(input_data)

        logger.info(f"🗺️ Planning for goal: '{goal_text}'")

        # 1. ゴールをトークナイズ
        try:
            tokens = self.tokenizer(goal_text, return_tensors="pt").input_ids.to(self.device)
            
            # 2. モデル推論 (スキル選択)
            # PlannerSNNは入力テキストから「最も適切なスキルカテゴリ」や「最初のアクション」を予測する
            with torch.no_grad():
                logits = self.model(tokens)
                probs = torch.softmax(logits, dim=-1)
                
                # 上位K個の候補を取得 (簡易的なプラン生成)
                top_k = 3
                values, indices = torch.topk(probs, k=top_k, dim=-1)
                
            # 3. プランの構築
            suggested_actions = []
            for i in range(top_k):
                # mypy fix: Explicit cast to int
                action_id = int(indices[0, i].item())
                confidence = values[0, i].item()
                action_name = self.action_space.get(action_id, f"Unknown_Skill_{action_id}")
                suggested_actions.append(f"{action_name} (Conf: {confidence:.2f})")

            # 簡易的なプラン: 最も確信度の高いアクションを採用しつつ、手順として提示
            # mypy fix: Explicit cast to int
            best_action_id = int(indices[0, 0].item())
            best_action_name = self.action_space.get(best_action_id, "Wait")
            
            plan = {
                "goal": goal_text,
                "suggested_actions": suggested_actions,
                "primary_action": best_action_name,
                "action_id": best_action_id,
                "status": "PLAN_READY"
            }
            
            return plan

        except Exception as e:
            logger.error(f"❌ Planning failed: {e}", exc_info=True)
            return {"error": str(e), "status": "PLAN_FAILED"}

    async def create_plan(self, high_level_goal: str) -> Plan:
        """
        EmergentCognitiveSystem 用の非同期メソッド。
        PlannerSNNの結果をラップして Plan オブジェクトとして返す。
        """
        # 計算自体は同期処理(process)を利用
        # 将来的にPlannerSNN自体が重い場合、run_in_executor等でラップする
        
        logger.info(f"🗺️ [Async] Creating plan for: {high_level_goal}")
        
        # 既存のロジックを再利用
        result_dict = self.process(high_level_goal)
        
        if result_dict.get("status") == "PLAN_FAILED":
            logger.warning("Planning failed, returning empty plan.")
            return Plan(goal=high_level_goal, task_list=[])

        # 結果をEmergentSystemが期待するタスクリスト形式に変換
        tasks = []
        primary_action = result_dict.get("primary_action", "Unknown")
        
        # シンプルな実装: プライマリアクションを実行するタスクを1つ生成
        # 実際にはここで複数のステップに分解するロジックが入る
        tasks.append({
            "description": f"Execute skill: {primary_action}",
            "expert_id": None, # エージェント探索に任せる
            "parameters": {
                "action_id": result_dict.get("action_id"),
                "confidence": result_dict.get("suggested_actions", [])
            }
        })
        
        return Plan(goal=high_level_goal, task_list=tasks)