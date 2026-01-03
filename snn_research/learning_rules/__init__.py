# ファイルパス: snn_research/learning_rules/__init__.py
# (更新)
# - ProbabilisticHebbian を追加
# - CausalTraceCreditAssignmentEnhancedV2 に対応
#
# 改善 (v2):
# - TripletSTDP を追加
#
# 改善 (v3):
# - BCMLearningRule を追加

from typing import Dict, Any
from .base_rule import BioLearningRule
from .stdp import STDP, TripletSTDP
from .reward_modulated_stdp import RewardModulatedSTDP
from .causal_trace import CausalTraceCreditAssignmentEnhancedV2
from .probabilistic_hebbian import ProbabilisticHebbian
# --- ▼ 修正 (v3) ▼ ---
from .bcm_rule import BCMLearningRule
# --- ▲ 修正 (v3) ▲ ---

def get_bio_learning_rule(name: str, params: Dict[str, Any]) -> BioLearningRule:
    """指定された名前に基づいて生物学的学習ルールオブジェクトを生成して返す。"""
    if name == "STDP":
        return STDP(**params.get('stdp', {}))
    elif name == "TRIPLET_STDP":
        # TripletSTDP は stdp と triplet_stdp の両方のパラメータ辞書をマージして受け取る
        stdp_params = params.get('stdp', {})
        triplet_params = params.get('triplet_stdp', {})
        
        # パラメータ名を TripletSTDP の __init__ に合わせる
        combined_params = {
            "learning_rate": stdp_params.get('learning_rate', 0.005),
            "a_plus_pair": stdp_params.get('a_plus', 1.0),
            "a_minus_pair": stdp_params.get('a_minus', 1.0),
            "tau_trace_pair": stdp_params.get('tau_trace', 20.0),
            "a_plus_triplet": triplet_params.get('a_plus_triplet', 0.01),
            "a_minus_triplet": triplet_params.get('a_minus_triplet', 0.001),
            "tau_trace_triplet": triplet_params.get('tau_trace_triplet', 50.0),
        }
        return TripletSTDP(**combined_params)
    # --- ▼ 修正 (v3) ▼ ---
    elif name == "BCM":
        return BCMLearningRule(**params.get('bcm', {}))
    # --- ▲ 修正 (v3) ▲ ---
    elif name == "REWARD_MODULATED_STDP":
        return RewardModulatedSTDP(**params.get('reward_modulated_stdp', {}))
    elif name == "CAUSAL_TRACE" or name == "CAUSAL_TRACE_ENHANCED" or name == "CAUSAL_TRACE_V2":
        # V1 (Enhanced) と V2 のパラメータ取得ロジックはほぼ同じだが、
        # V2固有のパラメータも考慮する
        stdp_params = params.get('stdp', {})
        causal_params = params.get('causal_trace', {}) # 設定キーは causal_trace のまま
        # Enhanced (V1) パラメータ
        enhanced_params = {
            'credit_time_decay': params.get('credit_time_decay', 0.95),
            'dynamic_lr_factor': params.get('dynamic_lr_factor', 2.0),
            'modulate_eligibility_tau': params.get('modulate_eligibility_tau', False),
            'min_eligibility_tau': params.get('min_eligibility_tau', 10.0),
            'max_eligibility_tau': params.get('max_eligibility_tau', 200.0)
        }
        # V2 パラメータ
        v2_params = {
            'context_modulation_strength': params.get('context_modulation_strength', 0.5),
            'competition_k_ratio': params.get('competition_k_ratio', 0.1),
            'rule_based_lr_factor': params.get('rule_based_lr_factor', 3.0)
        }
        # 全てのパラメータを結合
        combined_params = {**stdp_params, **causal_params, **enhanced_params, **v2_params}

        # 必須パラメータを確認 (RewardModulatedSTDPから継承)
        required = ['learning_rate', 'a_plus', 'a_minus', 'tau_trace', 'tau_eligibility']
        missing = [p for p in required if p not in combined_params]
        if missing:
            raise ValueError(f"Missing required parameters for CausalTraceV2: {', '.join(missing)}")

        return CausalTraceCreditAssignmentEnhancedV2(**combined_params) # クラス名を V2 に変更
    elif name == "PROBABILISTIC_HEBBIAN": # 新しいルールを追加
        return ProbabilisticHebbian(**params.get('probabilistic_hebbian', {}))
    else:
        raise ValueError(f"未知の学習ルール名です: {name}")

__all__ = [
    "BioLearningRule", "STDP", 
    "TripletSTDP",
    # --- ▼ 修正 (v3) ▼ ---
    "BCMLearningRule",
    # --- ▲ 修正 (v3) ▲ ---
    "RewardModulatedSTDP",
    "CausalTraceCreditAssignmentEnhancedV2",
    "ProbabilisticHebbian",
    "get_bio_learning_rule"
]