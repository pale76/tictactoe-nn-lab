"""グラフ作成（matplotlib）。

pyplot を使わず Figure を直接作るので、Gradio でもノートブックでもそのまま表示できる。
日本語フォントが見つからない環境では、ラベルを英語に自動で切り替える。
"""
from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.figure import Figure

JP_FONT_OK = False

_JP_FONT_CANDIDATES = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
                       "TakaoPGothic", "Yu Gothic", "Meiryo", "Hiragino Sans", "MS Gothic"]


def setup_japanese_font() -> bool:
    """日本語フォントを探して設定する。使えたら True。"""
    global JP_FONT_OK
    # 1) matplotlib-fontja（pip install matplotlib-fontja）があれば、それを使う
    try:
        import matplotlib_fontja  # noqa: F401
        if hasattr(matplotlib_fontja, "japanize"):
            matplotlib_fontja.japanize()
        JP_FONT_OK = True
        return True
    except Exception:
        pass
    # 2) 端末に入っている日本語フォントを探す
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in _JP_FONT_CANDIDATES:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                JP_FONT_OK = True
                return True
    except Exception:
        pass
    JP_FONT_OK = False
    return False


setup_japanese_font()


def _t(ja: str, en: str) -> str:
    return ja if JP_FONT_OK else en


def _class_names(short: bool = False):
    if short:
        return [_t("未決着\n/引分", "0: ongoing\n/draw"), _t("P1勝ち", "1: P1 win"),
                _t("P2勝ち", "2: P2 win")]
    return [_t("0 未決着/引き分け", "0 ongoing/draw"), _t("1 P1勝ち", "1 P1 win"),
            _t("2 P2勝ち", "2 P2 win")]


def _band(ax, x, mean, std, label, color):
    ax.plot(x, mean, label=label, color=color)
    if np.any(std > 0):
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def plot_history(multi, baseline=None) -> Figure:
    """エポックごとの正答率（左）と損失（右）。学習用とテスト用を並べる。"""
    fig = Figure(figsize=(11, 4.2))
    ax_acc, ax_loss = fig.subplots(1, 2)
    epochs = multi.epochs
    n_runs = len(multi.runs)

    for key, name, color in [("train_acc", _t("学習データ", "train"), "tab:blue"),
                             ("test_acc", _t("テストデータ", "test"), "tab:orange")]:
        mean, std = multi.curve(key)
        _band(ax_acc, epochs, mean * 100, std * 100, name, color)
    if baseline is not None:
        ax_acc.axhline(baseline * 100, color="gray", linestyle="--", linewidth=1,
                       label=_t("常に最多クラスと答える場合", "always-majority baseline"))
    ax_acc.set_title(_t(f"正答率（{n_runs}回の平均±標準偏差）", f"Accuracy (mean±std, {n_runs} runs)"))
    ax_acc.set_xlabel(_t("エポック（学習回数）", "epoch"))
    ax_acc.set_ylabel(_t("正答率 [%]", "accuracy [%]"))
    ax_acc.set_ylim(top=100.5)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="lower right")

    for key, name, color in [("train_loss", _t("学習データ", "train"), "tab:blue"),
                             ("test_loss", _t("テストデータ", "test"), "tab:orange")]:
        mean, std = multi.curve(key)
        _band(ax_loss, epochs, mean, std, name, color)
    ax_loss.set_title(_t("損失（小さいほど良い）", "Loss (lower is better)"))
    ax_loss.set_xlabel(_t("エポック（学習回数）", "epoch"))
    ax_loss.set_ylabel(_t("損失（クロスエントロピー）", "cross-entropy loss"))
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_confusion(multi) -> Figure:
    """テストデータの混同行列（行=正解, 列=NNの答え）。行ごとの割合を色と数字で表す。"""
    cm = multi.confusion_total().astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig = Figure(figsize=(5.2, 4.4))
    ax = fig.subplots()
    im = ax.imshow(ratio, vmin=0, vmax=1, cmap="Blues")
    names = _class_names(short=True)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel(_t("NNの答え", "predicted"))
    ax.set_ylabel(_t("正解（ルール）", "true (rule)"))
    ax.set_title(_t("混同行列（行ごとの割合）", "Confusion matrix (row-normalized)"))
    n_runs = len(multi.runs)
    for i in range(3):
        for j in range(3):
            avg = f"{cm[i, j] / n_runs:.1f}".replace(".0", "")  # 1回あたりの平均件数
            color = "white" if ratio[i, j] > 0.6 else "black"
            ax.text(j, i, f"{ratio[i, j] * 100:.1f}%\n({avg})", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_learning_curve(df, baseline=None) -> Figure:
    """学習データ数と正答率の関係（学習曲線）。df は train.learning_curve の戻り値。"""
    fig = Figure(figsize=(7, 4.4))
    ax = fig.subplots()
    x = df["n_train"].to_numpy()
    for prefix, name, color in [("train", _t("学習データ", "train"), "tab:blue"),
                                ("test", _t("テストデータ", "test"), "tab:orange")]:
        mean = df[f"{prefix}_acc_mean"].to_numpy() * 100
        std = df[f"{prefix}_acc_std"].to_numpy() * 100
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=name, color=color)
    if baseline is not None:
        ax.axhline(baseline * 100, color="gray", linestyle="--", linewidth=1,
                   label=_t("常に最多クラスと答える場合", "always-majority baseline"))
    if len(x) > 1 and x.max() / max(x.min(), 1) >= 20:
        ax.set_xscale("log")
    n_runs = int(df["n_runs"].iloc[0]) if len(df) else 0
    ax.set_title(_t(f"学習データ数と正答率（{n_runs}回の平均±標準偏差）",
                    f"Accuracy vs. training size (mean±std, {n_runs} runs)"))
    ax.set_xlabel(_t("学習データ数", "number of training samples"))
    ax.set_ylabel(_t("最終エポックの正答率 [%]", "final accuracy [%]"))
    ax.set_ylim(top=100.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
