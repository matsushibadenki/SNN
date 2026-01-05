# ファイルパス: snn_research/cognitive_architecture/tsetlin_machine.py
# 日本語タイトル: 簡易Tsetlin Machine (論理学習器)
# 機能説明: 
#   浮動小数点演算や誤差逆伝播を使わず、有限オートマトン(State Machine)を用いて
#   命題論理(AND/OR/NOT)を学習する。
#   GPU不要で、CPUやマイクロコントローラでの動作に最適化されている。

import numpy as np
import logging

logger = logging.getLogger(__name__)

class TsetlinMachine:
    """
    簡易的なTsetlin Machineの実装。
    二値分類問題に対して、論理的な特徴(Clause)を学習する。
    """
    def __init__(
        self, 
        number_of_clauses: int, 
        number_of_features: int, 
        states: int = 100, 
        s: float = 3.9, 
        threshold: int = 15
    ):
        """
        Args:
            number_of_clauses: 学習する論理式の数（偶数推奨: 半分はPositive, 半分はNegative用）
            number_of_features: 入力ビット数
            states: オートマトンの記憶容量（大きいほど記憶が強固）
            s: ノイズパラメータ（学習の探索範囲を制御）
            threshold: 決定閾値
        """
        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.states = states
        self.s = s
        self.threshold = threshold

        # TA (Tsetlin Automaton) の状態を初期化
        # 各リテラル(特徴そのもの + 否定)に対してオートマトンが存在
        # 状態: 1 〜 states*2
        # 1〜states: Exclude (論理式に含めない), states+1〜states*2: Include (含める)
        self.ta_states = np.random.choice(
            [states, states + 1], 
            size=(self.number_of_clauses, self.number_of_features, 2)
        ).astype(np.int32)

    def _calculate_clause_output(self, X: np.ndarray) -> np.ndarray:
        """
        現在のオートマトン状態で入力を評価し、各Clauseの出力(0 or 1)を計算する。
        Clauseは「リテラルのAND」で構成される。
        """
        # Include状態(> states)のリテラルのみを考慮
        # feature_on: 特徴そのまま, feature_off: 特徴の否定
        
        # X shape: (features,)
        # ta_states shape: (clauses, features, 2)
        
        # Action: 1 if state > states (Include), 0 otherwise
        action_include = (self.ta_states > self.states).astype(np.int32)
        
        clause_outputs = []
        for j in range(self.number_of_clauses):
            # Clause j の評価
            # 全リテラルに対して (Action=0 OR (Action=1 AND Input match)) が成立すれば1
            # Input match: k=0ならX[k]==1, k=1ならX[k]==0
            
            # リテラル0 (肯定): Include かつ X=0 なら False(0) -> 全体False
            # リテラル1 (否定): Include かつ X=1 なら False(0) -> 全体False
            
            # 簡略化: 全てのリテラル条件を満たしているかチェック
            # include_0 & (X==0) => Fail
            # include_1 & (X==1) => Fail
            
            fail_0 = (action_include[j, :, 0] == 1) & (X == 0)
            fail_1 = (action_include[j, :, 1] == 1) & (X == 1)
            
            if np.any(fail_0) or np.any(fail_1):
                clause_outputs.append(0)
            else:
                clause_outputs.append(1)
                
        return np.array(clause_outputs, dtype=np.int32)

    def predict(self, X: np.ndarray) -> int:
        """推論を実行"""
        clause_outputs = self._calculate_clause_output(X)
        
        # 前半のClauseはClass 1への投票(+1)、後半はClass 0への投票(-1)
        sum_votes = np.sum(clause_outputs[:self.number_of_clauses // 2]) - \
                    np.sum(clause_outputs[self.number_of_clauses // 2:])
        
        return 1 if sum_votes >= 0 else 0

    def fit(self, X: np.ndarray, y: int):
        """
        オンライン学習 (Type I / Type II Feedback)
        BPなし、確率的オートマトン更新のみ。
        """
        clause_outputs = self._calculate_clause_output(X)
        sum_votes = np.sum(clause_outputs[:self.number_of_clauses // 2]) - \
                    np.sum(clause_outputs[self.number_of_clauses // 2:])

        # 閾値による学習抑制 (マージン最大化に相当)
        if -self.threshold < sum_votes < self.threshold:
            # Type I Feedback (正解を強化)
            self._type_i_feedback(X, y, clause_outputs)
            # Type II Feedback (不正解を抑制)
            self._type_ii_feedback(X, y, clause_outputs)

    def _type_i_feedback(self, X: np.ndarray, y: int, clause_outputs: np.ndarray):
        """
        Type I: 正しい出力をしたClauseを強化し、より特定的にする(Includeへ誘導)
        または、誤って0を出したClauseにチャンスを与える。
        """
        for j in range(self.number_of_clauses):
            # 対象Clauseの決定 (Positive Classなら前半、Negative Classなら後半)
            is_positive_clause = (j < self.number_of_clauses // 2)
            target_clause = (y == 1 and is_positive_clause) or (y == 0 and not is_positive_clause)
            
            if target_clause:
                # 確率的に強化
                if np.random.rand() < (self.threshold - abs(sum(clause_outputs))) / (2 * self.threshold):
                    pass # ここでは簡易実装のためフィードバック確率調整は省略
                
                # Clauseが出力1だった場合: 有効なリテラルを強化(状態増)
                if clause_outputs[j] == 1:
                     for k in range(self.number_of_features):
                        # 肯定リテラル
                        if X[k] == 1: # 入力が1なら、肯定リテラルは正しい -> 状態+ (Include方向)
                            if self.ta_states[j, k, 0] < self.states * 2:
                                self.ta_states[j, k, 0] += 1
                        else: # 入力が0なら、肯定リテラルは邪魔 -> 状態- (Exclude方向)
                            if np.random.rand() < 1.0/self.s:
                                if self.ta_states[j, k, 0] > 1:
                                    self.ta_states[j, k, 0] -= 1
                        
                        # 否定リテラル (同様のロジック)
                        if X[k] == 0:
                            if self.ta_states[j, k, 1] < self.states * 2:
                                self.ta_states[j, k, 1] += 1
                        else:
                            if np.random.rand() < 1.0/self.s:
                                if self.ta_states[j, k, 1] > 1:
                                    self.ta_states[j, k, 1] -= 1

    def _type_ii_feedback(self, X: np.ndarray, y: int, clause_outputs: np.ndarray):
        """Type II: 誤って1を出力したClauseを弱体化(Excludeへ誘導)"""
        for j in range(self.number_of_clauses):
            is_positive_clause = (j < self.number_of_clauses // 2)
            target_clause = (y == 1 and not is_positive_clause) or (y == 0 and is_positive_clause)
            
            if target_clause and clause_outputs[j] == 1:
                # このClauseは誤ってActiveになった -> 状態を下げる
                for k in range(self.number_of_features):
                    # 肯定リテラルでExcludeでない(Input=0)ものは状態を下げる必要はない(元々Fail要因)
                    # Input=0 なのに Includeされているものは、このClauseをActiveにさせなかった要因(Good)
                    # ここではシンプルに「Activeになった」=「全てのIncludeリテラルがマッチした」なので、
                    # 全てのリテラルに対してExclude方向へ圧力をかける
                     if X[k] == 0 and self.ta_states[j, k, 0] > 1:
                         self.ta_states[j, k, 0] -= 1