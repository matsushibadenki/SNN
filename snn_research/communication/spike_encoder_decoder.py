# ファイルパス: snn_research/communication/spike_encoder_decoder.py
# (更新)
# Title: スパイク エンコーダー/デコーダー
# Description: ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、
#              抽象データ（テキスト、辞書）とスパイクパターンを相互に変換する。
# 修正点:
# - mypyエラー `Name "random" is not defined` を解消するため、randomモジュールをインポート。
# 改善点(v2): エージェント間通信プロトкоルの基礎として、メッセージに
#              「意図」と「内容」を含めるようにエンコード・デコード機能を拡張。
# 修正点(v3): mypyエラーを解消し、メソッド名をより汎用的に変更。

import torch
import json
import random
from typing import Dict, Any, Optional, Union

class SpikeEncoderDecoder:
    """
    テキストや辞書などの抽象データをスパイクパターンに変換し、
    またその逆の変換も行うクラス。
    """
    def __init__(self, num_neurons: int = 256, time_steps: int = 16):
        """
        Args:
            num_neurons (int): スパイク表現に使用するニューロン数。ASCII文字セットをカバーできる必要がある。
            time_steps (int): スパイクパターンの時間長。
        """
        self.num_neurons = num_neurons
        self.time_steps = time_steps

    def encode_data(self, data: Any) -> torch.Tensor:
        """
        任意のデータ（辞書、テキストなど）をJSON文字列に変換し、スパイクパターンにエンコードする。
        """
        try:
            json_str = json.dumps(data, sort_keys=True)
        except TypeError:
            json_str = json.dumps(str(data))
            
        spike_pattern = torch.zeros((self.num_neurons, self.time_steps))
        for char in json_str:
            neuron_id = ord(char) % self.num_neurons
            num_spikes = random.randint(1, 3)
            for _ in range(num_spikes):
                t = random.randint(0, self.time_steps - 1)
                spike_pattern[neuron_id, t] = 1
        return spike_pattern

    def decode_data(self, spikes: torch.Tensor) -> Any:
        """
        スパイクパターンをデコードして元のデータ（辞書、テキストなど）を復元する。
        """
        if spikes is None or not isinstance(spikes, torch.Tensor):
            return {"error": "Invalid spike pattern provided."}

        spike_counts = spikes.sum(dim=1)
        char_indices = torch.where(spike_counts > 0)[0]
        
        sorted_indices = sorted(char_indices, key=lambda idx: spike_counts[idx].item(), reverse=True)

        json_str = "".join([chr(int(idx)) for idx in sorted_indices if int(idx) < 256])
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json_str

