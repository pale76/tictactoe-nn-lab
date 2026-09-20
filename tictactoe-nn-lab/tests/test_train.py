"""学習まわりのテスト（PyTorch が必要。無い環境では自動でスキップ）。"""
import numpy as np
import pytest

pytest.importorskip("torch")

from tictactoe_nn.data import build_dataset  # noqa: E402
from tictactoe_nn.train import (TrainConfig, learning_curve, majority_baseline,  # noqa: E402
                                predict_proba, run_multiple, train_once)


def small_config(**kw):
    base = dict(epochs=3, n_train=200, hidden_layers=1, units=8, n_runs=2)
    base.update(kw)
    return TrainConfig(**base)


def test_train_once_shapes():
    ds = build_dataset("reachable")
    r = train_once(small_config(), ds, seed=0)
    assert r.history["epoch"] == [1, 2, 3]
    assert len(r.history["test_acc"]) == 3
    assert r.confusion.shape == (3, 3)
    assert r.confusion.sum() == ds.n_test
    assert r.n_train_used == 200


def test_same_seed_gives_same_result():
    ds = build_dataset("reachable")
    a = train_once(small_config(), ds, seed=1)
    b = train_once(small_config(), ds, seed=1)
    assert a.history["test_acc"] == b.history["test_acc"]
    assert a.history["train_loss"] == b.history["train_loss"]


def test_record_history_false_keeps_only_final_epoch():
    ds = build_dataset("reachable")
    r = train_once(small_config(), ds, seed=0, record_history=False)
    assert r.history["epoch"] == [3]


def test_n_train_is_clamped_to_pool():
    ds = build_dataset("reachable")
    r = train_once(small_config(n_train=10 ** 7, epochs=1), ds, seed=0)
    assert r.n_train_used == ds.n_pool


def test_run_multiple_and_dataframe():
    ds = build_dataset("reachable")
    progress = []
    multi = run_multiple(small_config(), ds, progress_callback=lambda d, t: progress.append((d, t)))
    assert len(multi.runs) == 2
    assert progress[-1] == (6, 6)
    df = multi.history_dataframe()
    assert len(df) == 2 * 3
    assert set(df["seed"]) == {0, 1}          # 実行ごとのseedが残っている
    assert "base_seed" in df.columns
    mean, std = multi.curve("test_acc")
    assert mean.shape == (3,) and std.shape == (3,)


def test_single_run_has_zero_std():
    ds = build_dataset("reachable")
    multi = run_multiple(small_config(n_runs=1), ds)
    _, std = multi.curve("test_acc")
    assert np.all(std == 0)


def test_hidden_layers_zero_and_all_activations_work():
    ds = build_dataset("reachable")
    train_once(small_config(hidden_layers=0, epochs=1), ds, seed=0)
    for act in ("relu", "tanh", "sigmoid", "leaky_relu"):
        train_once(small_config(activation=act, epochs=1), ds, seed=0)
    for rep in ("numeric", "onehot"):
        train_once(small_config(input_repr=rep, epochs=1), ds, seed=0)


def test_learning_curve_dataframe():
    ds = build_dataset("reachable")
    df = learning_curve(small_config(epochs=1, n_runs=1), [50, 100, 100, 10 ** 7], ds)
    assert list(df["n_train"]) == [50, 100, ds.n_pool]   # 重複は除去、上限は調整
    assert {"train_acc_mean", "test_acc_mean", "test_acc_std"} <= set(df.columns)


def test_model_learns_better_than_majority_baseline():
    ds = build_dataset("reachable")
    cfg = TrainConfig(epochs=30, n_train=3000, n_runs=1, hidden_layers=2, units=64,
                      input_repr="onehot")
    multi = run_multiple(cfg, ds)
    test_acc, _ = multi.final("test_acc")
    assert test_acc > majority_baseline(ds) + 0.05


def test_predict_proba_sums_to_one():
    ds = build_dataset("reachable")
    r = train_once(small_config(epochs=1), ds, seed=0)
    p = predict_proba(r.model, (0,) * 9, "onehot")
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-5
