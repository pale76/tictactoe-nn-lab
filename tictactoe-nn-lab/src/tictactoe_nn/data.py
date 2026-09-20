"""データセットの作成（学習/テストの分割）と、入力表現への変換。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .game import generate_all_boards, generate_reachable_boards, label_of

# 入力表現: 数値9個 / ワンホット27個
INPUT_REPRS = ("numeric", "onehot")
# 盤面の範囲: 実際に現れる盤面のみ / 全パターン
BOARD_SETS = ("reachable", "all")

N_CLASSES = 3


def input_dim(input_repr: str) -> int:
    if input_repr == "numeric":
        return 9
    if input_repr == "onehot":
        return 27
    raise ValueError(f"input_repr は {INPUT_REPRS} のどちらかです: {input_repr!r}")


def encode_boards(boards: np.ndarray, input_repr: str) -> np.ndarray:
    """盤面 (N, 9) の整数配列を、ニューラルネットワークへの入力 (N, 9 or 27) に変換する。

    numeric: 0/1/2 の値をそのまま 9 個の数値として使う。
    onehot : 各マスを (空, 〇, ×) の3値のワンホットにして 27 個にする。
    """
    boards = np.asarray(boards, dtype=np.int64)
    if boards.ndim == 1:
        boards = boards[None, :]
    if input_repr == "numeric":
        return boards.astype(np.float32)
    if input_repr == "onehot":
        onehot = np.eye(3, dtype=np.float32)[boards]  # (N, 9, 3)
        return onehot.reshape(len(boards), 27)
    raise ValueError(f"input_repr は {INPUT_REPRS} のどちらかです: {input_repr!r}")


@dataclass(eq=False)
class Dataset:
    """固定のテストデータと、学習データの候補（学習プール）。"""

    board_set: str
    train_boards: np.ndarray   # 学習プール (M, 9)
    train_labels: np.ndarray   # (M,)
    test_boards: np.ndarray    # テストデータ (K, 9) ※学習には一切使わない
    test_labels: np.ndarray    # (K,)

    @property
    def n_pool(self) -> int:
        """学習に使える最大件数。"""
        return len(self.train_boards)

    @property
    def n_test(self) -> int:
        return len(self.test_boards)


@lru_cache(maxsize=4)
def build_dataset(board_set: str = "reachable", test_ratio: float = 0.2,
                  split_seed: int = 42) -> Dataset:
    """盤面を生成し、ルールから正解ラベルを付けて、学習プールとテストデータに分ける。

    split_seed を固定しているので、実験条件を変えてもテストデータは常に同じになる。
    """
    if board_set == "reachable":
        boards = generate_reachable_boards()
    elif board_set == "all":
        boards = generate_all_boards()
    else:
        raise ValueError(f"board_set は {BOARD_SETS} のどちらかです: {board_set!r}")

    x = np.array(boards, dtype=np.int64)
    y = np.array([label_of(b) for b in boards], dtype=np.int64)

    perm = np.random.RandomState(split_seed).permutation(len(x))
    n_test = int(round(len(x) * test_ratio))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return Dataset(board_set, x[train_idx], y[train_idx], x[test_idx], y[test_idx])


def sample_train_subset(dataset: Dataset, n: int, seed: int):
    """学習プールから n 件を取り出す。

    同じ seed なら、n が違っても小さい方が大きい方の部分集合になる（入れ子）。
    学習データ数だけを変えた比較が公平になる。
    """
    n = max(1, min(int(n), dataset.n_pool))
    idx = np.random.RandomState(seed).permutation(dataset.n_pool)[:n]
    return dataset.train_boards[idx], dataset.train_labels[idx]


def class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=N_CLASSES)
