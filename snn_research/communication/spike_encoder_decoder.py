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
from typing import Any


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

    def latency_encode(self, data: Any) -> torch.Tensor:
        """
        データを決定論的なレイテンシコーディングでスパイクに変換する。
        ASCIIコードの値が小さいほど早いタイミング、大きいほど遅いタイミングでスパイクする。
        （より汎用的な実装では、値の重要度をタイミングにマップする）
        """
        try:
            json_str = json.dumps(data, sort_keys=True)
        except TypeError:
            json_str = json.dumps(str(data))

        spike_pattern = torch.zeros((self.num_neurons, self.time_steps))

        for i, char in enumerate(json_str):
            if i >= self.num_neurons:  # ニューロン数を超える文字数は切り捨て（またはチャンク分割が必要）
                break

            char_code = ord(char)
            # ASCIIコードを送信タイミングにマッピング (0-255 -> 0-time_steps)
            # 時間分解能に合わせてスケーリング
            timing = int(round((char_code / 255.0) * (self.time_steps - 1)))
            timing = max(0, min(self.time_steps - 1, timing))

            # ニューロンIDは文字の位置(i)に対応させる（空間配置）
            neuron_id = i % self.num_neurons
            spike_pattern[neuron_id, timing] = 1.0

        return spike_pattern

    def latency_decode(self, spikes: torch.Tensor) -> Any:
        """
        レイテンシコーディングされたスパイクパターンをデコードする。
        各ニューロンの最初のスパイク時刻から文字を復元する。
        """
        if spikes is None or not isinstance(spikes, torch.Tensor):
            return {"error": "Invalid spike pattern provided."}

        # 各ニューロンの発火タイミングを取得 (argmaxは最初の最大値=1のインデックスを返す)
        # 発火していない場合は0になるため、発火有無のマスクが必要
        has_spiked = (spikes.sum(dim=1) > 0)
        timings = spikes.argmax(dim=1)

        decoded_chars = []
        for i in range(self.num_neurons):
            if not has_spiked[i]:
                continue

            t = timings[i].item()
            # タイミングからASCIIコードを逆算
            # char_code = (t / (time_steps - 1)) * 255
            # 近似値になるため、完全に元の文字に戻らない可能性があるが、
            # ここでは簡易的に「タイミングビン」に割り当てられた代表文字とみなすか、
            # または単純に char_code = t * (255 / time_steps) とする

            # 修正: 文字自体がニューロンIDでなくタイミングで表現されると
            # 複数の文字が同じニューロン・違うタイミングで来る場合に混信する。
            # 今回のencode実装では「ニューロンID=文字位置」「タイミング=ASCII値」としているため、
            # 順序はニューロンID順で復元できる。

            estimated_char_code = int(
                round((t / (self.time_steps - 1)) * 255.0))
            # ASCII範囲内にクリップ
            estimated_char_code = max(
                32, min(126, estimated_char_code))  # 可読文字範囲
            decoded_chars.append(chr(estimated_char_code))

        # JSONとしての復元を試みる
        raw_str = "".join(decoded_chars)
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            return raw_str

    # Legacy interfaces kept for compatibility but redirected
    def encode_data(self, data: Any) -> torch.Tensor:
        return self.latency_encode(data)

    def decode_data(self, spikes: torch.Tensor) -> Any:
        # デコードは完全性が必要なため、従来のメソッドは使用不可（または近似復元）
        # レイテンシデコードの結果を返す
        return self.latency_decode(spikes)
