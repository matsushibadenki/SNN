# SNN4 プロジェクト：ANN に匹敵する AI への発展計画

## 1. 現状分析
SNN4 は「自己進化するデジタル生命体」を目指した包括的フレームワークであり、`train.py`、`run_distill_hpo.py` などを中心に学習・蒸留・進化の要素を備えている。

ただし、現段階では ANN 系の最新モデル（Transformer 系、ViT 系など）に比べると精度・学習安定性・GPU 最適化の面で課題がある。

---

## 2. 改善の方向性
ANN に匹敵する、あるいは超える性能を目指すための施策を 3 段階に分けて実施する。

### Step 1: Quick Wins（短期強化）
1. **PyTorch + snnTorch / Norse 化**  
   - GPU 上でサロゲート勾配学習を安定化。
   - CLI に `--backend {torch,snnTorch,norse}` オプションを追加。

2. **ANN→SNN 変換モジュールの実装**  
   - 変換手法（weight mapping + time coding 最適化）を導入。  
   - ImageNet/CIFAR 系で ANN 精度を再現。

3. **知識蒸留の強化**  
   - `run_distill_hpo.py` を拡張し、ANN 教師モデルを SNN 学習に統合。
   - 教師：ResNet/ViT、小規模タスクから実験開始。

---

### Step 2: Mid-term（構造・モデル改良）
4. **Spiking Transformer 系アーキテクチャ導入**  
   - 例：Spikformer, Spikingformer, Spike-driven Transformer。  
   - ImageNet レベルの精度を目標にする。

5. **ハイブリッド構成（ANN + SNN）**  
   - フロントエンドで ANN 特徴抽出、バックエンドで SNN 推論。  
   - 精度と効率の両立を狙う。

6. **ニューロンモデル・符号化改善**  
   - LIF に加え、適応閾値・多時間スケールモデルを導入。  
   - temporal / latency coding などを採用し、スパイク数削減。

---

### Step 3: Long-term（スケーラビリティ・展開）
7. **量子化・スパース化・構造的プルーニング**  
   - モデル効率化とエネルギー削減を両立。

8. **Neuromorphic ハードウェア対応**  
   - Intel Lava / Loihi / SpiNNaker などへ移植。  
   - エネルギー効率（J/inference）を指標に評価。

---

## 3. コード設計タスク
- `train.py` → snnTorch/Norse 対応。
- `scripts/ann2snn.py` → ANN→SNN 変換ロジック実装。
- `run_distill_hpo.py` → 教師モデル統合、ロス関数拡張。
- `configs/` → CIFAR-10 / DVS / ImageNet-subset 用設定追加。
- `benchmarks/` → 精度・スパイク数・レイテンシ・エネルギー記録。

---

## 4. 評価基準
| 指標 | 内容 |
|------|------|
| 精度 | Top-1 / Top-5 Accuracy |
| スパイク数 | 平均スパイク数（層別・推論ごと） |
| レイテンシ | 時間ステップ数・実時間換算（ms） |
| エネルギー | Joules/inference（推定または実測） |
| モデルサイズ | パラメータ・メモリ量 |

### タスク別評価
- **画像分類**：CIFAR-10/100, ImageNet-subset。
- **イベントカメラ**：N-MNIST, DVS128 Gesture。
- **RL・連続学習**：OpenAI Gym 環境でサンプル効率比較。

---

## 5. 優先実験案
1. **直接学習（snnTorch）** — CIFAR-10 でサロゲート勾配学習。  
2. **ANN→SNN 変換** — ResNet 変換と精度比較。  
3. **蒸留実験** — ANN 教師 → SNN 生徒で知識蒸留精度確認。

---

## 6. 管理・運用改善
- Issue テンプレート：`dataset, config, seed, commit_hash` 記録。
- 軽量 CI：MNIST 1epoch テストを自動化。
- Leaderboard：ベンチマーク結果を `benchmarks/` に公開。
- ドキュメント整備：設定例・学習レシピを README に追加。

---

## 7. リスクと戦略的落としどころ
- 現状、SNN が ANN に精度で完全勝利することは難しい。
- 短期目標は **精度を維持しつつスパイク効率・低電力で優位に立つ** こと。  
- 中長期では **SNN の時間情報活用・連続学習特性** により ANN を超える表現力を目指す。

---

## 8. 参考文献・実装リソース
- [snnTorch (PyTorch-based SNN library)](https://snntorch.readthedocs.io)
- [Norse (PyTorch extension for spiking models)](https://github.com/norse/norse)
- [Spikformer / Spikingformer / Spike-driven Transformer 系論文]
- [ANN→SNN 変換レビュー論文 2023-2024]
- [Intel Lava / Loihi / SpiNNaker ハードウェア]

---

## 9. 次の実務的アクション
- ✅ `train.py` を snnTorch ベースに置換する。
- ✅ `scripts/ann2snn.py` を新規作成し変換を実装。
- ✅ `run_distill_hpo.py` に教師モデル統合機能を追加。

いずれもモジュール構成を維持しつつ段階的に統合可能。

