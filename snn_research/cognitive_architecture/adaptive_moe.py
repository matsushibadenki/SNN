# snn_research/cognitive_architecture/adaptive_moe.py

from typing import List, Dict
import numpy as np
import time
import logging

# エキスパートのインターフェース定義（ダミー）


class Expert:
    def __init__(self, name: str):
        self.name = name


class LoRAAdapter:
    """LoRA (Low-Rank Adaptation) の簡易プレースホルダー"""

    def __init__(self, rank: int = 8):
        self.rank = rank
        self.weights = np.random.randn(rank, rank) * 0.01

    def apply(self, x: np.ndarray) -> np.ndarray:
        # 実際には行列演算を行う
        return x


class AdaptiveFrankenMoE:
    """学習可能なFrankenMoE - ユーザー適応型"""

    def __init__(self, base_experts: List[Expert]):
        self.base_experts = base_experts

        # ユーザーごとのLoRA層（軽量適応）
        self.user_adaptations: Dict[str, LoRAAdapter] = {}

        # エキスパート選択の学習履歴
        self.routing_history: List[Dict] = []

        # エキスパートごとの基礎信頼度 (0.0 - 1.0)
        self.expert_performance: Dict[str, float] = {
            expert.name: 0.5 for expert in base_experts
        }

        # キャッシュされたルーティングルール (Centroid -> Expert Name)
        self.routing_rules: Dict[int, str] = {}

        logging.info(
            f"AdaptiveFrankenMoE initialized with {len(base_experts)} experts.")

    def route_with_learning(self, query: str, user_id: str, spike_context: np.ndarray) -> List[Expert]:
        """学習可能なルーティング"""

        # 1. コンテキストから適切なエキスパートをスコアリング
        expert_scores = self._score_experts(query, spike_context)

        # 2. 過去の成功パターンを考慮してスコア調整
        adjusted_scores = self._adjust_by_history(expert_scores, query)

        # 3. 上位K個のエキスパートを選択 (K=2とする)
        top_k_experts = self._select_top_k(adjusted_scores, k=2)

        # 4. ユーザー適応層の準備 (この時点では適用せず、実行時に使用されることを想定)
        if user_id not in self.user_adaptations:
            # 新規ユーザー用のLoRA作成
            self.user_adaptations[user_id] = LoRAAdapter()

        return top_k_experts

    def _score_experts(self, query: str, spike_context: np.ndarray) -> Dict[str, float]:
        """クエリとコンテキストに基づいて各エキスパートをスコアリング"""
        scores = {}

        query_lower = query.lower()
        # context_intensity = np.mean(
        #     spike_context) if spike_context.size > 0 else 0.5

        for expert in self.base_experts:
            # ベースパフォーマンス + ランダムな揺らぎ（探索用）
            base_score = self.expert_performance.get(expert.name, 0.5)
            noise = np.random.normal(0, 0.05)

            # キーワードマッチング（修正版）
            keyword_bonus = 0.0

            # エキスパート名に含まれる単語がクエリにあるかチェック
            expert_terms = expert.name.replace("_", " ").split()
            for term in expert_terms:
                if term in ["expert"]:
                    continue
                if term in query_lower:
                    keyword_bonus += 0.3

            # 特定のキーワードに対するルール
            if "logic" in query_lower and "logical" in expert.name:
                keyword_bonus += 0.2
            if "reason" in query_lower and "logical" in expert.name:
                keyword_bonus += 0.2
            if "see" in query_lower or "look" in query_lower:
                if "visual" in expert.name:
                    keyword_bonus += 0.2

            scores[expert.name] = base_score + keyword_bonus + noise

        return scores

    def _adjust_by_history(self, scores: Dict[str, float], query: str) -> Dict[str, float]:
        """学習したルールに基づいてスコアを調整"""
        # クエリの簡易ハッシュからルールを検索
        query_hash = hash(query) % 1000

        if query_hash in self.routing_rules:
            preferred_expert = self.routing_rules[query_hash]
            if preferred_expert in scores:
                scores[preferred_expert] += 0.3  # 推奨エキスパートをより強くブースト

        return scores

    def _select_top_k(self, scores: Dict[str, float], k: int) -> List[Expert]:
        """スコア上位K個のエキスパートを選択"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_names = [item[0] for item in sorted_items[:k]]

        selected_experts = [
            e for e in self.base_experts if e.name in top_names]
        return selected_experts

    def learn_from_feedback(self, query: str, expert_name: str,
                            success: bool, spike_pattern: np.ndarray):
        """フィードバックから学習"""

        # ルーティング履歴を記録
        self.routing_history.append({
            'query': query,
            'expert': expert_name,
            'success': success,
            'spike_pattern': spike_pattern,
            'timestamp': time.time()
        })

        # エキスパート性能の更新（指数移動平均）
        alpha = 0.1
        current_perf = self.expert_performance.get(expert_name, 0.5)
        reward = 1.0 if success else 0.0

        new_perf = alpha * reward + (1 - alpha) * current_perf
        self.expert_performance[expert_name] = new_perf

        # 定期的にパターンを抽出してルール化
        if len(self.routing_history) % 10 == 0:  # 頻度を高めに設定
            self._extract_routing_patterns()

    def _extract_routing_patterns(self):
        """成功パターンからルールを抽出"""
        # 成功したケースのみ抽出
        successful = [h for h in self.routing_history if h['success']]
        if not successful:
            return

        # 簡易的なルール抽出
        for record in successful:
            query_hash = hash(record['query']) % 1000
            self.routing_rules[query_hash] = record['expert']

        logging.info(
            f"Updated routing rules. Total rules: {len(self.routing_rules)}")

    def optimize_routing(self):
        """睡眠時などに呼び出す全体最適化"""
        self._extract_routing_patterns()
        # 古い履歴のクリア
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
