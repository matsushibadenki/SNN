# ファイルパス: snn_research/core/adaptive_neuron_selector.py
# Title: 適応的ニューロンセレクタ (Adaptive Neuron Selector) - パラメータ引継ぎ強化版
# Description:
#   SNNの学習中の振る舞いを監視し、ニューロンのタイプを動的に切り替えるメタコントローラ。
#   修正点:
#   - ニューロン置換時に、古いニューロンから新しいニューロンへ重みパラメータ（weight, bias等）を
#     可能な限りコピーするように修正。これにより、切り替えによる急激な精度低下（忘却）を防ぐ。
#   - 形状が一致しないパラメータは無視し、警告をログ出力する。

import torch
import torch.nn as nn
from typing import List, Deque, Dict, Tuple, Type, cast, Any, Optional
from collections import deque
import logging

# 必要なニューロンクラスをインポート
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron, BistableIFNeuron

# ロガー設定
logger = logging.getLogger(__name__)

class AdaptiveNeuronSelector(nn.Module):
    """
    学習中の振る舞い（損失、スパイク率）を監視し、
    LIFとBIFニューロンなどを動的に切り替えるメタコントローラ。
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
            module_to_wrap (nn.Module): 内部のニューロンを切り替える対象のモジュール。
            layer_name_to_monitor (str): module_to_wrap内のニューロン層の名前。
            lif_params (Dict[str, Any]): LIFニューロンの初期化パラメータ。
            bif_params (Dict[str, Any]): BIFニューロンの初期化パラメータ。
            monitor_window (int): 監視ウィンドウサイズ。
            loss_plateau_threshold (float): 損失停滞の閾値。
            low_spike_rate_threshold (float): 低スパイク率の閾値。
            high_spike_rate_threshold (float): 高スパイク率の閾値。
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
        
        # 初期状態の確認
        try:
            self.monitored_neuron: nn.Module = self._find_layer(layer_name_to_monitor)
            self.current_neuron_type: Type[nn.Module] = type(self.monitored_neuron)
            logger.info(f"✅ AdaptiveNeuronSelectorが層 '{layer_name_to_monitor}' ({self.current_neuron_type.__name__}) の監視を開始しました。")
        except AttributeError:
            logger.error(f"❌ '{layer_name_to_monitor}' が 'module_to_wrap' に見つかりません。")
            self.monitored_neuron = nn.Identity() 
            self.current_neuron_type = nn.Identity

    def _find_layer(self, layer_name: str) -> nn.Module:
        """指定された名前のサブモジュールを見つける"""
        current_module: nn.Module = self.module_to_wrap
        for name in layer_name.split('.'):
            if not hasattr(current_module, name):
                raise AttributeError(f"モジュール '{type(current_module).__name__}' に属性 '{name}' が見つかりません。")
            current_module = getattr(current_module, name)
        return current_module

    def _transfer_weights(self, old_module: nn.Module, new_module: nn.Module) -> None:
        """
        古いモジュールから新しいモジュールへ、形状が一致するパラメータをコピーする。
        これにより、学習済みの特徴抽出能力を維持する。
        """
        old_state = old_module.state_dict()
        new_state = new_module.state_dict()
        
        transferred_keys = []
        
        for key, old_param in old_state.items():
            if key in new_state:
                new_param = new_state[key]
                if old_param.shape == new_param.shape:
                    with torch.no_grad():
                        new_param.copy_(old_param)
                    transferred_keys.append(key)
                else:
                    logger.debug(f"Skipping parameter '{key}': Shape mismatch ({old_param.shape} vs {new_param.shape})")
            else:
                logger.debug(f"Skipping parameter '{key}': Not in new module")
        
        if transferred_keys:
            logger.info(f"🔄 Transferred parameters: {transferred_keys}")
        else:
            logger.warning("⚠️ No parameters were transferred during neuron replacement. Weights are re-initialized.")

    def _replace_neuron_layer(self, target_class: Type[nn.Module], params: Dict[str, Any]) -> None:
        """監視対象のニューロン層を新しいクラスのインスタンスに置き換える"""
        if self.current_neuron_type == target_class:
            return

        try:
            # 元のニューロンの特徴量数を取得
            original_features: int = 0
            if hasattr(self.monitored_neuron, 'features'):
                original_features = cast(int, getattr(self.monitored_neuron, 'features'))
            elif hasattr(self.monitored_neuron, 'n_neurons'): 
                original_features = cast(int, getattr(self.monitored_neuron, 'n_neurons'))
            
            if original_features == 0:
                logger.warning(f"層 '{self.layer_name_to_monitor}' の特徴量数が取得できません。置き換えをスキップします。")
                return

            # 新しいニューロンインスタンスを作成
            params_no_features: Dict[str, Any] = params.copy()
            params_no_features.pop('features', None)
            
            new_neuron: nn.Module = target_class(features=original_features, **params_no_features)
            
            # デバイスを合わせる
            device_param = next(self.monitored_neuron.parameters(), None)
            if device_param is not None:
                original_device = device_param.device
                new_neuron.to(original_device)
            else:
                # パラメータがない場合（稀だが）はCPUデフォルトまたは前の親のデバイス
                pass

            # --- パラメータの引き継ぎ ---
            self._transfer_weights(self.monitored_neuron, new_neuron)

            # 親モジュールを取得して属性を置き換え
            parent_module: nn.Module = self.module_to_wrap
            layer_name_parts: List[str] = self.layer_name_to_monitor.split('.')
            if len(layer_name_parts) > 1:
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
        if hasattr(self.monitored_neuron, 'spikes'):
            spikes_tensor: Optional[torch.Tensor] = getattr(self.monitored_neuron, 'spikes')
            if spikes_tensor is not None and spikes_tensor.numel() > 0:
                spike_rate = spikes_tensor.mean().item()
        self.spike_rate_history.append(spike_rate)

        # 履歴が溜まるまで待機
        if len(self.loss_history) < self.monitor_window:
            return False, "Initializing history"

        # 2. 状態の診断
        avg_spike_rate: float = float(torch.tensor(list(self.spike_rate_history)).mean().item())
        loss_std_dev: float = float(torch.tensor(list(self.loss_history)).std().item())
        
        # 最近の平均損失
        recent_loss_mean = float(torch.tensor(list(self.loss_history)[-5:]).mean().item())

        # 判定ロジック
        if avg_spike_rate < self.low_spike_rate_threshold:
            # 症状: Dead Neuron (低発火)
            # 対策: BIFの双安定性またはノイズ耐性で活性化を試みる
            if self.current_neuron_type != BistableIFNeuron:
                self._replace_neuron_layer(BistableIFNeuron, self.bif_params)
                return True, f"Low spike rate ({avg_spike_rate:.3f} < {self.low_spike_rate_threshold}): Switched to BIF"
            
        elif avg_spike_rate > self.high_spike_rate_threshold:
            # 症状: Over-excitation (過剰発火)
            # 対策: 安定したLIFに戻す（閾値適応が働きやすい）
            if self.current_neuron_type != AdaptiveLIFNeuron:
                self._replace_neuron_layer(AdaptiveLIFNeuron, self.lif_params)
                return True, f"High spike rate ({avg_spike_rate:.3f} > {self.high_spike_rate_threshold}): Switched to LIF"

        elif loss_std_dev > self.loss_plateau_threshold * 5.0:
            # 症状: 学習が不安定・発散傾向
            if self.current_neuron_type != AdaptiveLIFNeuron:
                self._replace_neuron_layer(AdaptiveLIFNeuron, self.lif_params)
                return True, f"Loss unstable (std={loss_std_dev:.4f}): Switched to LIF for stability"

        elif loss_std_dev < self.loss_plateau_threshold and recent_loss_mean > 0.5:
            # 症状: 損失が高いまま停滞 (Local Minima)
            # 対策: BIFで表現力を高め、ダイナミクスを変える
            if self.current_neuron_type != BistableIFNeuron:
                self._replace_neuron_layer(BistableIFNeuron, self.bif_params)
                return True, f"Loss plateau (std={loss_std_dev:.4f}): Switched to BIF for exploration"
        
        return False, "Stable"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module_to_wrap(*args, **kwargs)
