# snn_research/cognitive_architecture/memory_consolidation.py
# Title: Hierarchical Memory System v1.0
# Description: 睡眠サイクルと記憶の固定化（Consolidation）を管理する統合システム。

from typing import Dict, List, Optional
import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field

# 既存モジュールのインポート（プロジェクト構造に基づく）
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus

logger = logging.getLogger(__name__)

@dataclass
class MemoryTrace:
    """記憶痕跡データクラス"""
    content: str
    spike_pattern: np.ndarray
    importance: float
    timestamp: float
    access_count: int = 0
    decay_rate: float = 0.05
    metadata: Dict = field(default_factory=dict)

class HierarchicalMemorySystem:
    """階層的記憶システム - 人間の脳を模倣した記憶管理"""
    
    def __init__(self, hippocampus: Optional[Hippocampus] = None, cortex: Optional[Cortex] = None):
        # レイヤー1: 作業記憶（コンテキストウィンドウ）
        self.working_memory: List[str] = []
        
        # レイヤー2: 短期記憶（海馬）- 既存モジュールがあればそれを使用
        self.hippocampus_module = hippocampus if hippocampus else Hippocampus()
        self.hippocampus_db: Dict[str, MemoryTrace] = {}
        
        # レイヤー3: 中期記憶 - エピソード記憶
        self.episodic_memory: List[Dict] = []
        
        # レイヤー4: 長期記憶（皮質）- SNNの重みおよびRAG
        self.cortex_module = cortex if cortex else Cortex()
        self.cortical_weights: Optional[np.ndarray] = None
        
        # レイヤー5: 知識グラフ（Cortex内のRAGSystemと統合）
        self.knowledge_graph = self.cortex_module.rag_system
        
        logger.info("🧠 HierarchicalMemorySystem initialized.")

    def store_experience(self, experience: Dict, spike_activity: np.ndarray):
        """経験を保存 - 重要度に応じて適切な階層へ振り分け"""
        importance = self._calculate_importance(experience, spike_activity)
        
        # 経験を一意のキーに変換
        exp_json = json.dumps(experience, ensure_ascii=False)
        key = self._generate_key(exp_json)
        
        trace = MemoryTrace(
            content=exp_json,
            spike_pattern=spike_activity,
            importance=importance,
            timestamp=time.time(),
            metadata=experience
        )
        
        # 重要度に応じた保存先決定
        if importance > 0.8:
            # 即座に長期記憶へ（高重要度）
            logger.info(f"⚡ Instant consolidation for high importance memory: {key[:8]}...")
            self._consolidate_to_cortex(trace)
        elif importance > 0.4:
            # 海馬（DB）で保持（中重要度）
            self.hippocampus_db[key] = trace
            # 既存のHippocampusモジュールとも同期（オプション）
            self.hippocampus_module.store_episode(experience)
        else:
            # エピソード記憶へ（低重要度、一時的）
            self.episodic_memory.append({
                'experience': experience,
                'spike_pattern': spike_activity,
                'timestamp': time.time()
            })

    def _generate_key(self, content: str) -> str:
        """コンテンツからハッシュキーを生成"""
        return str(hash(content))

    def _calculate_importance(self, experience: Dict, spikes: np.ndarray) -> float:
        """重要度計算 - スパイク同期性と新規性から判断"""
        # スパイク同期性（高いほど重要）
        synchrony = self._spike_synchrony(spikes)
        
        # 新規性（既存記憶との差異）- 簡易実装としてランダムまたはメタデータ依存
        novelty = self._calculate_novelty(experience)
        
        # 感情的価値（報酬系の活動）
        emotional_value = float(experience.get('reward', 0.0))
        
        # 重み付け平均
        score = 0.4 * synchrony + 0.4 * novelty + 0.2 * emotional_value
        return min(1.0, max(0.0, score))
    
    def _spike_synchrony(self, spikes: np.ndarray) -> float:
        """スパイク同期性の計算"""
        if spikes.size == 0:
            return 0.0
        # 時間軸方向の合計（各タイムステップの発火数）
        time_bins = spikes.sum(axis=0)  
        if time_bins.size == 0:
            return 0.0
        # 発火数の分散が大きい＝特定の瞬間に同期して発火している
        variance = np.var(time_bins)
        # 正規化（ヒューリスティックな値）
        return min(1.0, variance / 5.0)

    def _calculate_novelty(self, experience: Dict) -> float:
        """新規性の計算"""
        # 実際にはRAGの検索スコアの逆数などを使用するが、ここでは簡易実装
        # 'novelty'キーがあればそれを使用、なければデフォルト
        return float(experience.get('novelty', 0.5))

    def sleep_consolidation(self, duration_steps: int = 1000):
        """睡眠による記憶固定化 - 海馬→皮質への転送"""
        logger.info(f"💤 Sleep consolidation starting ({duration_steps} steps)...")
        
        # 海馬の記憶を重要度順にソート
        sorted_memories = sorted(
            self.hippocampus_db.items(),
            key=lambda x: x[1].importance,
            reverse=True
        )
        
        # 転送処理
        consolidation_threshold = 0.5
        transferred_count = 0
        
        for key, trace in sorted_memories:
            if trace.importance < consolidation_threshold:
                break
                
            # スパイクパターンをSTDP学習で重みに変換し、知識をRAGへ
            self._consolidate_to_cortex(trace)
            transferred_count += 1
            
            # 転送済みのため海馬から削除
            if key in self.hippocampus_db:
                del self.hippocampus_db[key]
        
        # 低重要度の記憶は忘却
        self._forget_low_importance_memories()
        
        logger.info(f"✅ Consolidation complete. Transferred: {transferred_count}")
    
    def _consolidate_to_cortex(self, trace: MemoryTrace):
        """STDPによる重み更新と概念の永続化"""
        # 1. 知識の保存 (Cortex RAG)
        try:
            content_dict = json.loads(trace.content)
            query = content_dict.get('query', '')
            response = content_dict.get('response', '')
            text_to_save = f"Q: {query}\nA: {response}"
            
            self.cortex_module.consolidate_episode(
                text_to_save, 
                source="sleep_consolidation"
            )
        except json.JSONDecodeError:
            self.cortex_module.consolidate_episode(trace.content)

        # 2. SNN重みの更新 (簡易STDP)
        if self.cortical_weights is None:
            # ニューロン数xニューロン数の行列を想定 (例: 256x256)
            dim = trace.spike_pattern.shape[0] if trace.spike_pattern.ndim > 0 else 256
            self.cortical_weights = np.random.randn(dim, dim) * 0.01
        
        delta_w = self._compute_stdp_update(trace.spike_pattern)
        
        # 学習率は重要度に比例
        learning_rate = 0.01 * trace.importance
        
        # サイズが合う場合のみ更新
        if delta_w.shape == self.cortical_weights.shape:
            self.cortical_weights += learning_rate * delta_w
    
    def _compute_stdp_update(self, spike_pattern: np.ndarray) -> np.ndarray:
        """STDP (Spike-Timing Dependent Plasticity) の簡易計算"""
        # spike_pattern: [Neurons, TimeSteps]
        if spike_pattern.ndim != 2:
            return np.zeros_like(self.cortical_weights) if self.cortical_weights is not None else np.zeros((256, 256))

        n_neurons, n_steps = spike_pattern.shape
        weight_update = np.zeros((n_neurons, n_neurons))
        
        # 簡易的なHebb則: 同時発火したペアの結合を強化
        # 本来はpre/postのタイミング差を見るが、ここでは相関行列で近似
        firing_rates = np.mean(spike_pattern, axis=1) # [Neurons]
        weight_update = np.outer(firing_rates, firing_rates)
        
        return weight_update

    def _forget_low_importance_memories(self):
        """忘却曲線に基づく記憶の削除"""
        current_time = time.time()
        keys_to_delete = []
        
        for key, trace in self.hippocampus_db.items():
            # 時間経過による減衰
            elapsed = current_time - trace.timestamp
            # 重要度が減衰する
            decayed_importance = trace.importance * np.exp(-trace.decay_rate * (elapsed / 3600.0)) # 1時間単位
            
            # 閾値を下回ったら忘却
            if decayed_importance < 0.2:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.hippocampus_db[key]
        
        if keys_to_delete:
            logger.info(f"🧹 Forgotten {len(keys_to_delete)} low-importance memories.")