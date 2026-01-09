# ファイルパス: snn_research/social/synesthetic_dialogue.py
# 日本語タイトル: Synesthetic Dialogue Manager (Grounding Game)
# 目的: 話し手(Speaker)と聞き手(Listener)の間で、視覚情報と言語情報の
#       相互変換（翻訳）ゲームを行い、シンボルグラウンディングを達成する。

import torch
import torch.nn.functional as F
from typing import Dict, Any

from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.social.communication_channel import CommunicationChannel

class SynestheticDialogue:
    def __init__(
        self,
        speaker: SynestheticAgent,
        listener: SynestheticAgent,
        channel: CommunicationChannel,
        vocab_size: int = 1000
    ):
        self.speaker = speaker
        self.listener = listener
        self.channel = channel
        self.vocab_size = vocab_size
        self.device = speaker.device

    def conduct_turn(self, target_image: torch.Tensor) -> Dict[str, Any]:
        """
        1ターンの対話を実行する。
        
        Flow:
        1. [Speaker]  画像を見る -> 言葉(Token)を発する
        2. [Channel]  言葉を伝送 (ノイズ付加)
        3. [Listener] 言葉を聞く -> 情景を想像(Dream/Latent)する
        4. [Judge]    Speakerが見た画像と、Listenerが想像した情景の一致度を判定
        
        Returns:
            metrics: {'similarity': float, 'message': str, ...}
        """
        batch_size = target_image.size(0)
        
        # --- 1. Speaker: See -> Speak ---
        # Brainを使って画像から思考（トークン列）を生成
        # SynestheticBrain.generate を利用
        # start_token_id = 1 (BOS) と仮定
        speaker_tokens = self.speaker.brain.generate(
            image_input=target_image, 
            start_token_id=1, 
            max_new_tokens=5
        )
        # generateはlist[int]を返す仕様(Brain v4)だが、バッチ処理のためTensor化が必要
        # ここでは簡易的にTensor変換 (B, Seq)
        speaker_tensor = torch.tensor(speaker_tokens, device=self.device).view(batch_size, -1)

        # --- 2. Channel: Transmit ---
        received_tokens = self.channel.transmit_tokens(speaker_tensor)

        # --- 3. Listener: Hear -> Dream (Imagine) ---
        # 言葉から視覚的イメージ（またはその潜在表現）を再構成する
        # ListenerのBrainでテキストをエンコードし、WorldModelのDecoderで映像化を試みる
        
        with torch.no_grad():
            # A. テキストのエンコード (Brain)
            # トークンID列をエンベディングに変換 (Brain内部のembedding層利用)
            listener_text_emb = self.listener.brain.core_brain.embedding(received_tokens) # (B, Seq, D)
            
            # B. 想像 (Dreaming / Cross-Modal Generation)
            # ここでは「言語コンテキスト」を「視覚コンテキスト」として解釈し、
            # World ModelのDecoder (Observation Reconstructor) に通すことで
            # 「言葉から連想される映像」を生成する。
            
            # Text Emb (B, Seq, D) -> Global Context (B, D)
            context_vector = listener_text_emb.mean(dim=1)
            
            # World ModelのVision Decoderを使用
            # (SpikingWorldModelは `decoders` 属性を持つ)
            if 'vision' in self.listener.world_model.decoders:
                imagined_image_feat = self.listener.world_model.decoders['vision'](context_vector)
            else:
                # Decoderがない場合は射影のみ (比較用)
                imagined_image_feat = context_vector 

        # --- 4. Evaluation (Grounding Check) ---
        # Speakerが見ている「真の画像特徴」と、Listenerが「想像した特徴」を比較
        
        # 真の画像特徴を取得 (Speaker's encoder output)
        with torch.no_grad():
            true_image_feat = self.speaker.brain.encoder.encode(target_image, modality='image')
            # 時間次元を平均化して特徴ベクトルにする
            if true_image_feat.dim() == 3:
                true_image_feat = true_image_feat.mean(dim=1)
            
            # 次元合わせ (Decoder出力とEncoder出力の次元が異なる場合の簡易補正)
            if imagined_image_feat.shape != true_image_feat.shape:
                # 線形補間等で合わせるか、射影層を通す必要があるが、
                # ここではコサイン類似度計算のため、同じ空間に射影されていると仮定(D_model統一)
                pass

        # 類似度計算 (Cosine Similarity)
        # 次元が一致している前提
        sim = F.cosine_similarity(imagined_image_feat, true_image_feat, dim=-1).mean()
        
        # 報酬 (Reward)
        # 類似度が高いほど高い報酬
        reward = sim.item()
        
        # 学習 (Communication Update)
        # Listenerは「聞いた言葉」と「正解画像(答え合わせ)」を使って、
        # "この言葉はこういう見た目だ" という結合(Cross-modal attention)を強化する
        self._update_agents(reward, received_tokens, target_image)

        return {
            'similarity': reward,
            'message': speaker_tokens,
            'received': received_tokens.tolist()
        }

    def _update_agents(self, reward: float, tokens: torch.Tensor, image: torch.Tensor):
        """
        コミュニケーションの成功度に基づいてエージェントを学習させる。
        """
        # 簡易実装: 成功時(reward > threshold)のみ、教師あり学習的に微調整
        if reward > 0.7: 
            # Listener: Image -> Text の結びつきを強化 (逆も然り)
            # ここではBrainのトレーニングモードへの切り替え等は省略し、
            # 概念的な「強化」ステップとする。
            pass