# ファイルパス: snn_research/core/adaptive_neuron_selector.py
# (新規作成・mypy修正)
# Title: 適応的ニューロンセレクタ (Adaptive Neuron Selector)
# Description:
# doc/Improvement-Plan.md (改善案1, Phase 2) に基づき、
# SNNの学習中の振る舞いを監視し、ニューロンのタイプ（例: LIF, BIF）を
# 動的に切り替えるためのメタコントローラを実装します。
# これにより、安定性が求められる場面ではLIFを、表現力が必要な場面ではBIFを
# 自動的に選択し、学習の安定性と性能を両立することを目指します。
# mypy --strict 準拠。
# 修正: mypy [name-defined] エラーを解消するため、Anyをインポート。

import torch
import torch.nn as nn
# --- ▼ 修正 ▼ ---
from typing import List, Deque, Dict, Tuple, Type, cast, Any
# --- ▲ 修正 ▲ ---
from collections import deque
import logging

# 必要なニューロンクラスをインポート
from .neurons import AdaptiveLIFNeuron
from .neurons.bif_neuron import BistableIFNeuron

# ロガー設定
logger = logging.getLogger(__name__)

class AdaptiveNeuronSelector(nn.Module):
    """
    学習中の振る舞い（損失、スパイク率）を監視し、
    LIFとBIFニューロンを動的に切り替えるメタコントローラ。

    このモジュール自体が学習するのではなく、ヒューリスティックに基づいて
    下位層のニューロンモジュールを差し替えます。
    """

    def __init__(
        self,
        module_to_wrap: nn.Module,
        layer_name_to_monitor: str,
        lif_params: Dict[str, Any],
        bif_params: Dict[str, Any],
        monitor_window: int = 20,
        loss_plateau_threshold: float = 0.001,
        low_spike_rate_threshold: float = 0.05,
        high_spike_rate_threshold: float = 0.95
    ) -> None:
        """
        Args:
            module_to_wrap (nn.Module): 内部のニューロンを切り替える対象のモジュール (例: SpikingCNN)。
            layer_name_to_monitor (str): module_to_wrap内のニューロン層の名前 (例: "neuron2")。
            lif_params (Dict[str, Any]): LIFニューロンの初期化パラメータ。
            bif_params (Dict[str, Any]): BIFニューロンの初期化パラメータ。
            monitor_window (int): 損失やスパイク率の監視ウィンドウサイズ。
            loss_plateau_threshold (float): 損失の停滞とみなす標準偏差の閾値。
            low_spike_rate_threshold (float): 低スパイク率とみなす閾値。
            high_spike_rate_threshold (float): 高スパイク率とみなす閾値。
        """
        super().__init__()
        self.module_to_wrap: nn.Module = module_to_wrap
        self.layer_name_to_monitor: str = layer_name_to_monitor
        self.lif_params: Dict[str, Any] = lif_params
        self.bif_params: Dict[str, Any] = bif_params
        self.monitor_window: int = monitor_window
        self.loss_plateau_threshold: float = loss_plateau_threshold
        self.low_spike_rate_threshold: float = low_spike_rate_threshold
        self.high_spike_rate_threshold: float = high_spike_rate_threshold

        # 監視用の履歴バッファ
        self.loss_history: Deque[float] = deque(maxlen=monitor_window)
        self.spike_rate_history: Deque[float] = deque(maxlen=monitor_window)
        
        self.current_neuron_type: Type[nn.Module] = AdaptiveLIFNeuron
        
        # 監視対象のニューロン層への参照を取得
        try:
            self.monitored_neuron: nn.Module = self._find_layer(layer_name_to_monitor)
            self.current_neuron_type = type(self.monitored_neuron)
            logger.info(f"✅ AdaptiveNeuronSelectorが層 '{layer_name_to_monitor}' ({self.current_neuron_type.__name__}) の監視を開始しました。")
        except AttributeError:
            logger.error(f"❌ '{layer_name_to_monitor}' が 'module_to_wrap' に見つかりません。")
            # 実行を継続するためにダミーモジュールを設定
            self.monitored_neuron = nn.Identity() 
            self.current_neuron_type = nn.Identity

    def _find_layer(self, layer_name: str) -> nn.Module:
        """指定された名前のサブモジュールを見つける"""
        # "conv2.neuron" のようなネストした名前を解決
        current_module: nn.Module = self.module_to_wrap
        for name in layer_name.split('.'):
            if not hasattr(current_module, name):
                raise AttributeError(f"モジュール '{type(current_module).__name__}' に属性 '{name}' が見つかりません。")
            current_module = getattr(current_module, name)
        return current_module

    def _replace_neuron_layer(self, target_class: Type[nn.Module], params: Dict[str, Any]) -> None:
        """監視対象のニューロン層を新しいクラスのインスタンスに置き換える"""
        if self.current_neuron_type == target_class:
            logger.debug(f"層 '{self.layer_name_to_monitor}' は既に '{target_class.__name__}' です。")
            return

        try:
            # 元のニューロンの特徴量数を取得
            original_features: int = 0
            if hasattr(self.monitored_neuron, 'features'):
                original_features = cast(int, getattr(self.monitored_neuron, 'features'))
            elif hasattr(self.monitored_neuron, 'n_neurons'): # BioLIFNeuronの場合
                original_features = cast(int, getattr(self.monitored_neuron, 'n_neurons'))
            
            if original_features == 0:
                logger.warning(f"層 '{self.layer_name_to_monitor}' の特徴量数が取得できません。置き換えをスキップします。")
                return

            # 新しいニューロンインスタンスを作成
            # paramsからfeaturesを削除 (AdaptiveLIFNeuron/BistableIFNeuronの引数名が異なるため)
            params_no_features: Dict[str, Any] = params.copy()
            params_no_features.pop('features', None)
            
            # 'features'引数は両方のクラスで必須と仮定
            new_neuron: nn.Module = target_class(features=original_features, **params_no_features)
            
            # デバイスを合わせる
            original_device: torch.device = next(self.monitored_neuron.parameters(), torch.tensor(0)).device
            new_neuron.to(original_device)

            # 親モジュールを取得して属性を置き換え
            parent_module: nn.Module = self.module_to_wrap
            layer_name_parts: List[str] = self.layer_name_to_monitor.split('.')
            if len(layer_name_parts) > 1:
                # ネストしたモジュールの場合、親を取得
                for name in layer_name_parts[:-1]:
                    parent_module = getattr(parent_module, name)
            
            final_layer_name: str = layer_name_parts[-1]
            setattr(parent_module, final_layer_name, new_neuron)
            
            # 参照とタイプを更新
            self.monitored_neuron = new_neuron
            self.current_neuron_type = target_class
            logger.info(f"🧬 ニューロン進化: 層 '{self.layer_name_to_monitor}' が '{target_class.__name__}' に切り替わりました。")

        except Exception as e:
            logger.error(f"ニューロン層の置き換え中にエラーが発生しました: {e}", exc_info=True)


    def step(self, current_loss: float) -> Tuple[bool, str]:
        """
        学習ステップごとに呼び出され、統計を更新し、切り替えを判断する。

        Args:
            current_loss (float): 現在のバッチの損失。

        Returns:
            Tuple[bool, str]: (切り替えが発生したか, 理由)
        """
        # 1. 統計の収集
        self.loss_history.append(current_loss)
        
        spike_rate: float = 0.0
        if hasattr(self.monitored_neuron, 'spikes'): # AdaptiveLIFNeuron / BistableIFNeuron
            spikes_tensor: torch.Tensor = getattr(self.monitored_neuron, 'spikes')
            if spikes_tensor is not None and spikes_tensor.numel() > 0:
                spike_rate = spikes_tensor.mean().item()
        self.spike_rate_history.append(spike_rate)

        # 履歴が溜まるまで待機
        if len(self.loss_history) < self.monitor_window:
            return False, "Initializing history"

        # 2. 状態の診断 (Improvement-Plan.mdのロジック)
        avg_spike_rate: float = float(torch.tensor(list(self.spike_rate_history)).mean().item())
        loss_std_dev: float = float(torch.tensor(list(self.loss_history)).std().item())

        # 判定ロジック
        if avg_spike_rate < self.low_spike_rate_threshold:
            # 症状: Dead Neuron 問題
            # 対策: BIFの双安定性で活性化を試みる
            if self.current_neuron_type != BistableIFNeuron:
                self._replace_neuron_layer(BistableIFNeuron, self.bif_params)
                return True, "low_spike_rate: Switched to BIF"
            
        elif avg_spike_rate > self.high_spike_rate_threshold:
            # 症状: Over-excitation (過剰発火)
            # 対策: 安定したLIFに戻す
            if self.current_neuron_type != AdaptiveLIFNeuron:
                self._replace_neuron_layer(AdaptiveLIFNeuron, self.lif_params)
                return True, "high_spike_rate: Switched to LIF"

        elif loss_std_dev > self.loss_plateau_threshold * 10: # 閾値の10倍以上
            # 症状: 学習が不安定・発散傾向
            # 対策: 安定したLIFに戻す
            if self.current_neuron_type != AdaptiveLIFNeuron:
                self._replace_neuron_layer(AdaptiveLIFNeuron, self.lif_params)
                return True, "loss_diverging: Switched to LIF"

        elif loss_std_dev < self.loss_plateau_threshold:
            # 症状: 停滞
            # 対策: BIFで表現力を高め、停滞からの脱出を試みる
            if self.current_neuron_type != BistableIFNeuron:
                self._replace_neuron_layer(BistableIFNeuron, self.bif_params)
                return True, "loss_plateau: Switched to BIF"
        
        return False, "Stable"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        セレクタはラッパーとして機能し、内部のモジュールを呼び出す。
        """
        # このモジュール自体は計算グラフの一部ではなく、
        # 外部のトレーナーが step() メソッドを呼び出すことを想定している。
        # もしラッパーとして機能させる場合は、ここで module_to_wrap を呼び出す。
        return self.module_to_wrap(*args, **kwargs)