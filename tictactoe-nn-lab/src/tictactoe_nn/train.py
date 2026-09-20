"""学習・評価・複数回実行・学習曲線。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import (Dataset, N_CLASSES, build_dataset, encode_boards,
                   sample_train_subset)
from .model import MLP, count_parameters


@dataclass
class TrainConfig:
    """実験条件。画面から変えられる項目はここにまとまっている。"""

    epochs: int = 50            # 学習回数（全学習データを何周するか）
    n_train: int = 3000         # 学習データ数（学習プールの件数を超えたら自動で上限に調整）
    hidden_layers: int = 2      # 隠れ層の数（0で隠れ層なし）
    units: int = 64             # 隠れ層1つあたりのユニット数
    activation: str = "relu"    # relu / tanh / sigmoid / leaky_relu
    lr: float = 1e-3            # 学習率
    input_repr: str = "onehot"  # numeric(9個) / onehot(27個)
    batch_size: int = 64
    n_runs: int = 5             # 同じ条件で繰り返す回数（初期値を変えて平均を取る）
    seed: int = 0               # 1回目の乱数シード（i回目は seed+i）
    board_set: str = "reachable"  # reachable(5,478通り) / all(19,683通り)


@dataclass
class RunResult:
    """1回分の学習結果。"""

    seed: int
    n_train_used: int
    history: dict          # epoch / train_loss / test_loss / train_acc / test_acc のリスト
    confusion: np.ndarray  # テストデータの混同行列 (行=正解, 列=NNの答え)
    model: nn.Module


@dataclass
class MultiRunResult:
    """同じ条件で複数回学習した結果。"""

    config: TrainConfig
    n_train_used: int
    n_test: int
    runs: list = field(default_factory=list)

    @property
    def epochs(self) -> np.ndarray:
        return np.array(self.runs[0].history["epoch"])

    def curve(self, key: str):
        """key の履歴について、実行回数をまたいだ (平均, 標準偏差) を返す。"""
        arr = np.array([r.history[key] for r in self.runs], dtype=float)
        return arr.mean(axis=0), _std(arr, axis=0)

    def final(self, key: str):
        """最終エポックの値の (平均, 標準偏差)。"""
        vals = np.array([r.history[key][-1] for r in self.runs], dtype=float)
        return float(vals.mean()), float(_std(vals))

    def confusion_total(self) -> np.ndarray:
        return np.sum([r.confusion for r in self.runs], axis=0)

    def per_class_accuracy(self) -> np.ndarray:
        """クラス別正答率（そのクラスが正解の盤面のうち、NNが当てた割合）。"""
        cm = self.confusion_total().astype(float)
        totals = cm.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(totals > 0, np.diag(cm) / totals, np.nan)
        return acc

    def history_dataframe(self) -> pd.DataFrame:
        """エポックごとの履歴（実行回数 × エポック）。CSV保存用。"""
        frames = []
        for i, r in enumerate(self.runs):
            df = pd.DataFrame(r.history)
            df.insert(0, "seed", r.seed)
            df.insert(0, "run", i + 1)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        for k, v in _config_columns(self.config, self.n_train_used).items():
            df[k] = v
        return df


def _std(arr, axis=None):
    """標準偏差。1回しか実行していないときは 0 にする。"""
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0] if axis == 0 else arr.size
    if n <= 1:
        return np.zeros(arr.shape[1:]) if axis == 0 else 0.0
    return arr.std(axis=axis, ddof=1)


def _config_columns(config: TrainConfig, n_train_used: int) -> dict:
    cols = asdict(config)
    cols["n_train_requested"] = cols.pop("n_train")
    cols["n_train_used"] = n_train_used
    cols["base_seed"] = cols.pop("seed")  # 実行ごとの seed 列と名前が重ならないようにする
    return cols


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn):
    """損失・正答率・予測を返す（学習はしない）。"""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return loss, acc, pred


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def train_once(config: TrainConfig, dataset: Dataset, seed: int,
               record_history: bool = True,
               epoch_callback: Optional[Callable[[int], None]] = None) -> RunResult:
    """1回分の学習。

    record_history=False のときは最終エポックだけ評価する（学習曲線の高速化用）。
    epoch_callback は、1エポック終わるたびに epoch 番号を渡して呼ばれる。
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    train_boards, train_labels = sample_train_subset(dataset, config.n_train, seed)
    x_train = torch.from_numpy(encode_boards(train_boards, config.input_repr))
    y_train = torch.from_numpy(train_labels.astype(np.int64))
    x_test = torch.from_numpy(encode_boards(dataset.test_boards, config.input_repr))
    y_test = torch.from_numpy(dataset.test_labels.astype(np.int64))

    model = MLP(config.input_repr, config.hidden_layers, config.units, config.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    n = len(x_train)
    batch_size = max(1, int(config.batch_size))
    epochs = int(config.epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

        if record_history or epoch == epochs:
            tr_loss, tr_acc, _ = _evaluate(model, x_train, y_train, loss_fn)
            te_loss, te_acc, _ = _evaluate(model, x_test, y_test, loss_fn)
            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)
        if epoch_callback is not None:
            epoch_callback(epoch)

    _, _, pred = _evaluate(model, x_test, y_test, loss_fn)
    confusion = _confusion_matrix(dataset.test_labels, pred.numpy())
    return RunResult(seed=seed, n_train_used=n, history=history,
                     confusion=confusion, model=model)


def run_multiple(config: TrainConfig, dataset: Optional[Dataset] = None,
                 record_history: bool = True,
                 progress_callback: Optional[Callable[[int, int], None]] = None) -> MultiRunResult:
    """同じ条件で n_runs 回学習する。i回目の乱数シードは config.seed + i。

    progress_callback(done_epochs, total_epochs) が、1エポックごとに呼ばれる。
    """
    if dataset is None:
        dataset = build_dataset(config.board_set)
    n_runs = max(1, int(config.n_runs))
    total = n_runs * int(config.epochs)

    result = MultiRunResult(config=config, n_train_used=min(int(config.n_train), dataset.n_pool),
                            n_test=dataset.n_test)
    for i in range(n_runs):
        offset = i * int(config.epochs)
        cb = None
        if progress_callback is not None:
            cb = (lambda epoch, offset=offset: progress_callback(offset + epoch, total))
        run = train_once(config, dataset, seed=config.seed + i,
                         record_history=record_history, epoch_callback=cb)
        result.runs.append(run)
    return result


def learning_curve(config: TrainConfig, sizes, dataset: Optional[Dataset] = None,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """学習データ数を変えながら学習し、最終エポックの正答率を比べる。

    注意: エポック数は固定なので、データが少ないほど重み更新の回数も少なくなる。
    """
    if dataset is None:
        dataset = build_dataset(config.board_set)
    sizes = sorted({max(1, min(int(s), dataset.n_pool)) for s in sizes})
    total = len(sizes) * max(1, int(config.n_runs)) * int(config.epochs)

    rows = []
    done_before = 0
    for size in sizes:
        cfg = replace(config, n_train=size)
        cb = None
        if progress_callback is not None:
            cb = (lambda done, _total, base=done_before: progress_callback(base + done, total))
        multi = run_multiple(cfg, dataset, record_history=False, progress_callback=cb)
        done_before += max(1, int(config.n_runs)) * int(config.epochs)
        tr_m, tr_s = multi.final("train_acc")
        te_m, te_s = multi.final("test_acc")
        row = {"n_train": multi.n_train_used, "n_runs": len(multi.runs),
               "train_acc_mean": tr_m, "train_acc_std": tr_s,
               "test_acc_mean": te_m, "test_acc_std": te_s}
        row.update({k: v for k, v in _config_columns(config, multi.n_train_used).items()
                    if k not in ("n_train_requested", "n_train_used")})
        rows.append(row)
    return pd.DataFrame(rows)


def majority_baseline(dataset: Dataset) -> float:
    """「常に最も多いクラスと答える」だけの場合のテスト正答率。

    クラスに偏りがあるため、NNの正答率がこの値を大きく超えていないと、
    「学習できた」とは言えない。
    """
    counts = np.bincount(dataset.test_labels, minlength=N_CLASSES)
    return float(counts.max() / counts.sum())


def predict_proba(model: nn.Module, board, input_repr: str) -> np.ndarray:
    """1つの盤面について、3クラスの確率 [未決着/引き分け, P1勝ち, P2勝ち] を返す。"""
    x = torch.from_numpy(encode_boards(np.array(board, dtype=np.int64), input_repr))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return probs.numpy()


def describe_model(config: TrainConfig) -> str:
    """モデル構成の1行説明（例: 入力27 → [64×2, relu] → 3）。"""
    model = MLP(config.input_repr, config.hidden_layers, config.units, config.activation)
    dim = 27 if config.input_repr == "onehot" else 9
    if int(config.hidden_layers) == 0:
        body = "隠れ層なし"
    else:
        body = f"{int(config.units)}ユニット × {int(config.hidden_layers)}層, {config.activation}"
    return f"入力{dim} → [{body}] → 3（パラメータ数 {count_parameters(model):,}）"
