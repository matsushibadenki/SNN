# snn_research/learning_rules/__init__.py
# Title: Learning Rules Package Init (Fixed Exports)
# Description:
#   - ProbabilisticHebbian を公開し、ImportErrorを修正。
#   - get_bio_learning_rule ファクトリ関数を更新。

from typing import Dict, Any
from .base_rule import BioLearningRule
from .stdp import STDP, TripletSTDP
from .reward_modulated_stdp import RewardModulatedSTDP
from .causal_trace import CausalTraceCreditAssignmentEnhancedV2
from .probabilistic_hebbian import ProbabilisticHebbian

# エイリアス (後方互換性のため)
CausalTraceCreditAssignment = CausalTraceCreditAssignmentEnhancedV2

def get_bio_learning_rule(name: str, params: Dict[str, Any]) -> BioLearningRule:
    """
    学習ルール名からインスタンスを生成して返すファクトリ関数。
    """
    name = name.lower()
    
    if name == 'stdp':
        return STDP(**params.get('stdp', {}))
    
    elif name == 'triplet_stdp':
        return TripletSTDP(**params.get('stdp', {}))
    
    elif name == 'reward_modulated_stdp':
        return RewardModulatedSTDP(**params.get('reward_modulated_stdp', {}))
    
    elif name in ['causal_trace', 'causal_trace_enhanced', 'causal_trace_v2']:
        combined_params = params.get('reward_modulated_stdp', {}).copy()
        combined_params.update(params.get('causal', {}))
        # 必要なパラメータが不足している場合のデフォルト値補完などはここで行う
        return CausalTraceCreditAssignmentEnhancedV2(**combined_params)
    
    elif name == 'probabilistic_hebbian':
        return ProbabilisticHebbian(**params.get('probabilistic_hebbian', {}))
        
    else:
        # デフォルトはSTDP
        return STDP()

__all__ = [
    "BioLearningRule",
    "STDP",
    "TripletSTDP",
    "RewardModulatedSTDP",
    "CausalTraceCreditAssignmentEnhancedV2",
    "CausalTraceCreditAssignment",
    "ProbabilisticHebbian",
    "get_bio_learning_rule"
]