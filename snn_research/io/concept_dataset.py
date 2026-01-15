# snn_research/io/concept_dataset.py
# ファイルパス: snn_research/io/concept_dataset.py
# 日本語タイトル: 概念拡張データセット (Concept Augmented Dataset)
# 修正: mypyの型エラー (int | float mismatch) を修正。

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional, cast


class ConceptAugmentedDataset(Dataset):
    """
    既存のデータセット（MNIST, CIFARなど）をラップし、
    画像データと共に、そのクラスに対応する「抽象概念リスト」を返します。
    """

    def __init__(self, base_dataset: Dataset, concept_map: Dict[int, List[str]]):
        """
        Args:
            base_dataset: PyTorchのデータセット（画像, ラベル を返すもの）
            concept_map: クラスID(int)をキーとし、概念単語リスト(List[str])を値とする辞書
        """
        self.base_dataset = base_dataset
        self.concept_map = concept_map
        self.default_concept = ["unknown", "object"]

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[str], int]:
        """
        Returns:
            image (Tensor): 画像データ
            concepts (List[str]): 画像に対応する抽象概念のリスト
            label (int): 正解ラベル
        """
        # ベースデータセットから画像とラベルを取得
        item = self.base_dataset[idx]
        img = item[0]
        label = item[1]

        # ラベルを確実に int に変換 (mypy対策)
        label_idx: int
        if isinstance(label, torch.Tensor):
            label_idx = int(label.item())
        else:
            label_idx = int(label)

        concepts = self.concept_map.get(label_idx, self.default_concept)

        # imgはTensorと仮定して返す
        return img, concepts, label_idx


def create_mnist_concepts() -> Dict[int, List[str]]:
    """MNISTデータセット向けの概念定義サンプル"""
    return {
        0: ["round", "closed_loop", "void", "symmetry"],
        1: ["vertical", "straight", "single", "line"],
        2: ["curve", "horizontal", "bottom_line", "swan_shape"],
        3: ["curve", "double_loop", "open_left", "wavy"],
        4: ["crossing", "vertical", "horizontal", "triangle_top"],
        5: ["horizontal_top", "vertical_left", "curve_bottom", "angular"],
        6: ["loop_bottom", "curve_top", "enclosed", "whistle"],
        7: ["horizontal_top", "diagonal", "sharp", "angle"],
        8: ["double_loop", "symmetry", "infinity", "closed"],
        9: ["loop_top", "diagonal", "vertical", "balloon"]
    }


def create_cifar10_concepts() -> Dict[int, List[str]]:
    """CIFAR-10データセット向けの概念定義サンプル"""
    return {
        0: ["vehicle", "flight", "sky", "wings", "machine"],       # airplane
        1: ["vehicle", "land", "wheels", "metal", "transport"],    # automobile
        2: ["animal", "feathers", "wings", "beak", "flight"],      # bird
        3: ["animal", "fur", "pet", "whiskers", "agile"],          # cat
        4: ["animal", "wild", "antlers", "brown", "forest"],       # deer
        5: ["animal", "pet", "loyal", "bark", "fur"],              # dog
        6: ["animal", "amphibian", "green", "jump", "pond"],       # frog
        7: ["animal", "mammal", "large", "ride", "mane"],          # horse
        8: ["vehicle", "water", "float", "ocean", "transport"],    # ship
        9: ["vehicle", "wheels", "cargo", "heavy", "transport"]    # truck
    }
