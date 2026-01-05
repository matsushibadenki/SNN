# ファイルパス: snn_research/core/factories.py
# 日本語タイトル: ニューロン・ファクトリ
# ファイルの目的・内容:
#   ニューロンモデルの生成ロジックを集約し、各モデルクラスでのコード重複を排除する。
#   将来的なニューロンタイプの追加に対して、このファイルを修正するだけで対応可能にする（Open-Closed Principle）。

import torch.nn as nn
from typing import Dict, Any, Type, List
import logging

from snn_research.core.neurons import (
    AdaptiveLIFNeuron,
    IzhikevichNeuron,
    GLIFNeuron,
    TC_LIF,
    DualThresholdNeuron,
    ScaleAndFireNeuron,
    BistableIFNeuron,
    EvolutionaryLeakLIF,
    ProbabilisticLIFNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder

logger = logging.getLogger(__name__)

class NeuronFactory:
    """
    設定辞書に基づいて適切なニューロンインスタンスを生成するファクトリクラス。
    パラメータのフィルタリングを自動化し、モデル実装側の負担を軽減する。
    """

    # ニューロンタイプ名とクラスのマッピング
    _NEURON_REGISTRY: Dict[str, Type[nn.Module]] = {
        'lif': AdaptiveLIFNeuron,
        'adaptive_lif': AdaptiveLIFNeuron,
        'izhikevich': IzhikevichNeuron,
        'glif': GLIFNeuron,
        'tc_lif': TC_LIF,
        'dual_threshold': DualThresholdNeuron,
        'scale_and_fire': ScaleAndFireNeuron,
        'bif': BistableIFNeuron,
        'evolutionary_leak_lif': EvolutionaryLeakLIF,
        'probabilistic_lif': ProbabilisticLIFNeuron,
        'dttfs': DifferentiableTTFSEncoder, # 特殊ケース
    }

    # 各ニューロンクラスが受け入れるパラメータのホワイトリスト
    # (将来的にinspectモジュールで動的に取得することも可能だが、明示的な定義が安全)
    _PARAM_WHITELIST: Dict[Type[nn.Module], List[str]] = {
        AdaptiveLIFNeuron: [
            'features', 'tau_mem', 'base_threshold', 'adaptation_strength',
            'target_spike_rate', 'noise_intensity', 'threshold_decay',
            'threshold_step', 'v_reset', 'homeostasis_rate', 'refractory_period'
        ],
        IzhikevichNeuron: ['features', 'a', 'b', 'c', 'd', 'dt'],
        GLIFNeuron: ['features', 'base_threshold', 'gate_input_features'],
        TC_LIF: [
            'features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init',
            'base_threshold', 'v_reset'
        ],
        DualThresholdNeuron: [
            'features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset'
        ],
        ScaleAndFireNeuron: ['features', 'num_levels', 'base_threshold'],
        BistableIFNeuron: [
            'features', 'v_threshold_high', 'v_reset', 'tau_mem', 'bistable_strength',
            'v_rest', 'unstable_equilibrium_offset'
        ],
        EvolutionaryLeakLIF: [
            'features', 'initial_tau', 'v_threshold', 'v_reset', 'detach_reset',
            'learn_threshold'
        ],
        ProbabilisticLIFNeuron: [
            'features', 'tau_mem', 'threshold', 'temperature', 'noise_intensity', 'v_reset'
        ],
        DifferentiableTTFSEncoder: [
            'num_neurons', 'duration', 'initial_sensitivity'
        ]
    }

    @classmethod
    def get_neuron_class(cls, type_name: str) -> Type[nn.Module]:
        """タイプ名からクラスを取得"""
        if type_name not in cls._NEURON_REGISTRY:
            logger.warning(f"Unknown neuron type '{type_name}'. Falling back to 'lif'.")
            return AdaptiveLIFNeuron
        return cls._NEURON_REGISTRY[type_name]

    @classmethod
    def filter_params(cls, neuron_class: Type[nn.Module], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたニューロンクラスに必要なパラメータのみを抽出する。
        """
        whitelist = cls._PARAM_WHITELIST.get(neuron_class, [])
        if not whitelist:
            # ホワイトリストが未定義の場合は、最低限のパラメータのみ通す（安全策）
            logger.warning(f"No parameter whitelist found for {neuron_class.__name__}. Using default filtering.")
            return {k: v for k, v in config.items() if k in ['features', 'tau_mem', 'base_threshold', 'v_reset']}
        
        return {k: v for k, v in config.items() if k in whitelist}

    @classmethod
    def create(cls, features: int, config: Dict[str, Any], **override_params: Any) -> nn.Module:
        """
        ニューロンインスタンスを生成する。
        
        Args:
            features (int): ニューロン数（特徴量次元）。
            config (Dict[str, Any]): ニューロン設定辞書（'type' キーを含む）。
            **override_params: configの内容を上書きするパラメータ。
        """
        # 設定のコピーと上書き
        full_config = config.copy()
        full_config.update(override_params)
        
        neuron_type = full_config.pop('type', 'lif')
        neuron_class = cls.get_neuron_class(neuron_type)
        
        # 特殊なパラメータマッピング
        if neuron_class == DifferentiableTTFSEncoder:
             # DTTFSは 'features' ではなく 'num_neurons' を使う場合があるが、
             # 統一インターフェースとして features を受け取り、内部で変換する
             full_config['num_neurons'] = features
        elif neuron_class == GLIFNeuron:
             if 'gate_input_features' not in full_config:
                 full_config['gate_input_features'] = features
        else:
             full_config['features'] = features

        # パラメータフィルタリング
        filtered_params = cls.filter_params(neuron_class, full_config)
        
        try:
            return neuron_class(**filtered_params)
        except TypeError as e:
            logger.error(f"Failed to instantiate {neuron_class.__name__} with params: {filtered_params}. Error: {e}")
            # フォールバック: デフォルトLIF
            return AdaptiveLIFNeuron(features=features)