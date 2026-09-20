import numpy as np
import pytest

from tictactoe_nn.data import (build_dataset, class_counts, encode_boards, input_dim,
                               sample_train_subset)


def test_split_sizes_and_no_overlap():
    ds = build_dataset("reachable")
    assert ds.n_test == round(5478 * 0.2)
    assert ds.n_pool + ds.n_test == 5478
    train = {tuple(b) for b in ds.train_boards}
    test = {tuple(b) for b in ds.test_boards}
    assert not (train & test)          # テストデータは学習プールに含まれない
    assert len(train) == ds.n_pool     # 重複なし


def test_all_boards_split():
    ds = build_dataset("all")
    assert ds.n_pool + ds.n_test == 3 ** 9


def test_split_is_fixed_across_calls():
    a = build_dataset("reachable", 0.2, 42)
    b = build_dataset("reachable", 0.2, 42)
    assert np.array_equal(a.test_boards, b.test_boards)


def test_all_classes_present_in_test():
    ds = build_dataset("reachable")
    assert (class_counts(ds.test_labels) > 0).all()


def test_encode_shapes_and_values():
    boards = np.array([[0, 1, 2, 2, 1, 0, 0, 1, 0]])
    assert encode_boards(boards, "numeric").shape == (1, 9)
    onehot = encode_boards(boards, "onehot")
    assert onehot.shape == (1, 27)
    assert onehot.sum() == 9                      # 各マスで1つだけ1
    assert onehot[0, 0:3].tolist() == [1, 0, 0]   # 1マス目は空
    assert onehot[0, 3:6].tolist() == [0, 1, 0]   # 2マス目は〇
    assert onehot[0, 6:9].tolist() == [0, 0, 1]   # 3マス目は×
    assert input_dim("numeric") == 9 and input_dim("onehot") == 27


def test_encode_single_board_1d():
    assert encode_boards(np.zeros(9, dtype=int), "onehot").shape == (1, 27)


def test_invalid_repr():
    with pytest.raises(ValueError):
        encode_boards(np.zeros((1, 9), dtype=int), "foo")


def test_train_subsets_are_nested_and_clamped():
    ds = build_dataset("reachable")
    small, _ = sample_train_subset(ds, 100, seed=3)
    large, _ = sample_train_subset(ds, 500, seed=3)
    assert np.array_equal(small, large[:100])
    huge, _ = sample_train_subset(ds, 10 ** 9, seed=3)
    assert len(huge) == ds.n_pool
