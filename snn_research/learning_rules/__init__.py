# snn_research/learning_rules/__init__.py
# 修正: ProbabilisticHebbian を正しいモジュールからインポート

from typing import Dict, Any
from .base_rule import BioLearningRule
from .reward_modulated_stdp import RewardModulatedSTDP, EmotionModulatedSTDP
from .causal_trace import CausalTraceCreditAssignmentEnhancedV2 
from .probabilistic_hebbian import ProbabilisticHebbian

def get_bio_learning_rule(rule_name: str = "reward_modulated_stdp", config: Dict[str, Any] = {}, **kwargs: Any) -> BioLearningRule:
    """
    Args:
        rule_name: 学習則の名前 (位置引数またはキーワード引数)
        config: 設定辞書
        **kwargs: その他の引数 (nameなど) を吸収
    """
    # kwargsにnameが含まれている場合、それを優先
    if 'name' in kwargs:
        rule_name = kwargs['name']
        
    params = config.get("params", {})

    if rule_name == "reward_modulated_stdp":
        return RewardModulatedSTDP(**params)
    
    elif rule_name == "emotion_modulated_stdp":
        return EmotionModulatedSTDP(**params)
    
    elif rule_name == "causal_trace" or rule_name == "bio_causal_sparse":
        # paramsを結合
        combined = {**params, **config.get("causal_trace", {})}
        return CausalTraceCreditAssignmentEnhancedV2(**combined)
        
    elif rule_name == "probabilistic_hebbian":
        return ProbabilisticHebbian(**params)
    
    else:
        return RewardModulatedSTDP()