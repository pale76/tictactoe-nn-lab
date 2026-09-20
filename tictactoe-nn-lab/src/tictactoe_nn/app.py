"""Gradio による Web 画面。

起動方法:
    python src/main.py                  # ローカル（通常はこちらを使う）
    （Colab では notebooks/colab_demo.ipynb から launch() を呼ぶ）
"""
from __future__ import annotations

import base64
import random
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from .data import build_dataset, sample_train_subset
from .game import LABEL_NAMES, board_to_text, is_reachable, label_of
from .plots import plot_confusion, plot_history, plot_learning_curve
from .train import (TrainConfig, describe_model, learning_curve, majority_baseline,
                    predict_proba, run_multiple)

# リポジトリ直下（src/tictactoe_nn/app.py の2つ上）。ロゴ・ファビコンの場所を探すのに使う
REPO_ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = REPO_ROOT / "assets" / "images" / "logo.png"
FAVICON_PATH = REPO_ROOT / "assets" / "icons" / "favicon.png"

# ---------------------------------------------------------------- 選択肢

BOARD_SET_CHOICES = [("実際に現れる盤面のみ（5,478通り）", "reachable"),
                     ("全パターン（19,683通り・ありえない盤面を含む）", "all")]
INPUT_REPR_CHOICES = [("ワンホット27個（各マスを 空/〇/× の3値で表す）", "onehot"),
                      ("数値9個（0/1/2 をそのまま入力）", "numeric")]
ACTIVATION_CHOICES = ["relu", "tanh", "sigmoid", "leaky_relu"]
LR_CHOICES = [("0.0001", 1e-4), ("0.0003", 3e-4), ("0.001", 1e-3), ("0.003", 3e-3),
              ("0.01", 1e-2), ("0.03", 3e-2), ("0.1", 1e-1)]
BATCH_CHOICES = [16, 32, 64, 128, 256]
CELL_CHOICES = [("空", 0), ("〇", 1), ("×", 2)]
CELL_TEXT_TO_INT = {"空": 0, "〇": 1, "×": 2}

DEFAULT_SIZES = "50, 100, 200, 500, 1000, 2000, 4000"


# ---------------------------------------------------------------- 補助関数

def _build_config(board_set, input_repr, epochs, n_train, hidden_layers, units,
                  activation, lr, batch_size, n_runs, seed) -> TrainConfig:
    return TrainConfig(
        epochs=int(epochs), n_train=int(n_train), hidden_layers=int(hidden_layers),
        units=int(units), activation=str(activation), lr=float(lr),
        input_repr=str(input_repr), batch_size=int(batch_size), n_runs=int(n_runs),
        seed=int(seed or 0), board_set=str(board_set))


def _make_progress(progress, desc):
    """学習の進捗を Gradio の進捗バーに渡す（更新回数が多すぎないよう間引く）。"""
    def callback(done, total):
        step = max(1, total // 100)
        if done % step == 0 or done == total:
            progress((done, total), desc=desc)
    return callback


def _save_csv(df: pd.DataFrame, prefix: str) -> str:
    out_dir = Path(tempfile.mkdtemp(prefix="tictactoe_nn_"))
    path = out_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")  # Excelで開いても文字化けしない
    return str(path)


def _parse_sizes(text: str) -> list:
    parts = [p for p in text.replace("、", ",").replace(" ", ",").split(",") if p]
    try:
        sizes = [int(p) for p in parts]
    except ValueError:
        raise ValueError("学習データ数は、カンマ区切りの整数で入力してください（例: 50, 100, 500）")
    if not sizes or min(sizes) < 1:
        raise ValueError("学習データ数は1以上の整数を1つ以上入力してください")
    if len(sizes) > 12:
        raise ValueError("学習データ数の候補は12個までにしてください")
    return sizes


def _condition_text(config: TrainConfig, n_used: int) -> str:
    """学習条件の1行説明（タブ③で、どのモデルかを示すのに使う）。"""
    repr_name = "ワンホット27個" if config.input_repr == "onehot" else "数値9個"
    board_name = "実際に現れる盤面" if config.board_set == "reachable" else "全パターン"
    if int(config.hidden_layers) == 0:
        net = "隠れ層なし"
    else:
        net = f"隠れ層{int(config.hidden_layers)}層×{int(config.units)}({config.activation})"
    return (f"{repr_name}・{config.epochs}エポック・学習データ{n_used:,}件・{net}・"
            f"学習率{config.lr}・{board_name}・seed {config.seed}")


def _membership_text(board, state) -> str:
    """その盤面が、判定に使うモデルの学習データ／テストデータのどちらに入っているか。"""
    if board in state["train_set"]:
        return "このモデルの**学習に使われた**盤面（覚えているだけかもしれません）"
    if board in state["test_set"]:
        return "**テストデータ**の盤面（学習に使っていない、初めて見る盤面）"
    if board in state["pool_set"]:
        return "学習に使える盤面のうち、**今回は使われなかった**盤面（初めて見る盤面）"
    return "今回のデータセットに**含まれない**盤面"


def _cell_to_int(value) -> int:
    if isinstance(value, str) and value in CELL_TEXT_TO_INT:
        return CELL_TEXT_TO_INT[value]
    return int(value)


def _pm(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% ± {std * 100:.1f}"


def _summary_markdown(multi, dataset) -> str:
    cfg = multi.config
    tr_m, tr_s = multi.final("train_acc")
    te_m, te_s = multi.final("test_acc")
    base = majority_baseline(dataset)
    lines = [
        "### 結果（最終エポック）",
        f"- **テストデータの正答率: {_pm(te_m, te_s)}**（{len(multi.runs)}回の平均 ± 標準偏差）",
        f"- 学習データの正答率: {_pm(tr_m, tr_s)}",
        f"- 学習とテストの差: {(tr_m - te_m) * 100:+.1f} ポイント（大きいほど過学習の傾向）",
        f"- 参考: 「常に最多クラスと答える」だけでも {base * 100:.1f}%"
        "（NNがこれを大きく超えていれば、盤面と勝敗の関係を学べている）",
        f"- モデル: {describe_model(cfg)}",
        f"- 学習データ {multi.n_train_used:,}件 / テストデータ {multi.n_test:,}件"
        f"（学習に使える上限 {dataset.n_pool:,}件）",
        f"- 条件: {cfg.epochs}エポック, 学習率 {cfg.lr}, バッチ {cfg.batch_size}, "
        f"seed {cfg.seed}〜{cfg.seed + len(multi.runs) - 1}",
    ]
    if int(cfg.n_train) > dataset.n_pool:
        lines.append(f"- ※指定した学習データ数 {int(cfg.n_train):,} は上限を超えたため、"
                     f"{dataset.n_pool:,}件に調整しました")
    lines.append("- 下の「盤面を判定」タブで、1回目に学習したモデルを試せます")
    return "\n".join(lines)


def _per_class_dataframe(multi, dataset) -> pd.DataFrame:
    acc = multi.per_class_accuracy()
    counts = np.bincount(dataset.test_labels, minlength=3)
    rows = []
    for k in range(3):
        rows.append({"ラベル": k, "内容": LABEL_NAMES[k], "テスト件数": int(counts[k]),
                     "正答率(%)": "-" if np.isnan(acc[k]) else f"{acc[k] * 100:.1f}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 画面から呼ばれる処理

def on_run_experiment(board_set, input_repr, epochs, n_train, hidden_layers, units,
                      activation, lr, batch_size, n_runs, seed,
                      progress=gr.Progress()):
    config = _build_config(board_set, input_repr, epochs, n_train, hidden_layers, units,
                           activation, lr, batch_size, n_runs, seed)
    dataset = build_dataset(config.board_set)
    progress(0, desc="学習を開始します")
    multi = run_multiple(config, dataset,
                         progress_callback=_make_progress(progress, "学習中"))
    baseline = majority_baseline(dataset)
    summary = _summary_markdown(multi, dataset)
    fig_history = plot_history(multi, baseline)
    fig_confusion = plot_confusion(multi)
    per_class = _per_class_dataframe(multi, dataset)
    csv_path = _save_csv(multi.history_dataframe(), "history")
    train_boards, _ = sample_train_subset(dataset, config.n_train, config.seed)  # 1回目と同じ選び方
    state = {"model": multi.runs[0].model, "input_repr": config.input_repr,
             "board_set": config.board_set,
             "condition": _condition_text(config, multi.n_train_used),
             "train_set": {tuple(int(v) for v in b) for b in train_boards},
             "test_set": {tuple(int(v) for v in b) for b in dataset.test_boards},
             "pool_set": {tuple(int(v) for v in b) for b in dataset.train_boards}}
    return summary, fig_history, fig_confusion, per_class, csv_path, state


def on_run_learning_curve(board_set, input_repr, epochs, n_train, hidden_layers, units,
                          activation, lr, batch_size, n_runs, seed, sizes_text,
                          progress=gr.Progress()):
    try:
        sizes = _parse_sizes(sizes_text)
    except ValueError as e:
        raise gr.Error(str(e))
    config = _build_config(board_set, input_repr, epochs, n_train, hidden_layers, units,
                           activation, lr, batch_size, n_runs, seed)
    dataset = build_dataset(config.board_set)
    progress(0, desc="学習を開始します")
    df = learning_curve(config, sizes, dataset,
                        progress_callback=_make_progress(progress, "学習曲線を計算中"))
    baseline = majority_baseline(dataset)
    fig = plot_learning_curve(df, baseline)

    table = pd.DataFrame({
        "学習データ数": df["n_train"],
        "学習データの正答率(%)": [f"{m * 100:.1f} ± {s * 100:.1f}"
                            for m, s in zip(df["train_acc_mean"], df["train_acc_std"])],
        "テストデータの正答率(%)": [f"{m * 100:.1f} ± {s * 100:.1f}"
                             for m, s in zip(df["test_acc_mean"], df["test_acc_std"])],
    })
    used_max = int(df["n_train"].max())
    summary = (
        "### 学習曲線\n"
        f"- 学習データ数を {', '.join(str(int(v)) for v in df['n_train'])} に変えて、"
        f"各 {int(df['n_runs'].iloc[0])} 回学習した最終エポックの正答率です\n"
        f"- 学習に使える上限は {dataset.n_pool:,}件（超える指定は上限に調整）\n"
        f"- 参考: 「常に最多クラスと答える」だけでも {baseline * 100:.1f}%\n"
        f"- エポック数は {config.epochs} で固定のため、データが少ないほど重みの更新回数も少なくなります"
        f"（最大 {used_max:,}件）")
    csv_path = _save_csv(df, "learning_curve")
    return summary, fig, table, csv_path


def on_judge(c0, c1, c2, c3, c4, c5, c6, c7, c8, state):
    board = tuple(_cell_to_int(c) for c in (c0, c1, c2, c3, c4, c5, c6, c7, c8))
    rule = label_of(board)
    lines = ["```", board_to_text(board), "```",
             f"- ルール上の正解: **{rule}（{LABEL_NAMES[rule]}）**"]
    if not is_reachable(board):
        lines.append("- ※この盤面は、プレイヤー1先手の実際のゲームでは現れません"
                     "（〇と×の数が不自然、勝敗が付いた後にも打っている、など）。"
                     "実際に現れる盤面だけで学習したNNでは、答えが当てにならないことがあります")
    if not state:
        lines.append("- NNの答え: まだ学習していません。先に「① 学習実験」タブで学習してください")
        return "\n".join(lines), None
    probs = predict_proba(state["model"], board, state["input_repr"])
    pred = int(np.argmax(probs))
    mark = "✅ 正解" if pred == rule else "❌ 不正解"
    lines.append(f"- NNの答え: **{pred}（{LABEL_NAMES[pred]}）** … {mark}")
    lines.append(f"- この盤面は: {_membership_text(board, state)}")
    lines.append(f"- 判定に使ったモデル（繰り返しの1回目）: {state['condition']}")
    label_probs = {f"{k}: {LABEL_NAMES[k]}": float(probs[k]) for k in range(3)}
    return "\n".join(lines), label_probs


def on_random_board(board_set):
    """テストデータ（学習に使っていない盤面）からランダムに1つ選ぶ。"""
    dataset = build_dataset(board_set or "reachable")
    board = random.choice(list(dataset.test_boards))
    return [int(v) for v in board]


def on_clear_board():
    return [0] * 9


# ---------------------------------------------------------------- 画面の組み立て

HEADER_TITLE = "〇×ゲーム ニューラルネットワーク実験ラボ"
HEADER_TEXT = ("盤面（9マス）から勝敗を当てるニューラルネットワークを学習させ、"
               "学習回数・学習データ数・ネットワーク構成による正答率の変化を調べます。"
               "左で条件を決めて、右のタブで実行してください。")


def _build_header():
    """ロゴ付きのヘッダー。ロゴを読めない場合は、文字だけのヘッダーにする。"""
    try:
        logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        gr.HTML(
            '<div style="display:flex;align-items:center;gap:14px;">'
            f'<img src="data:image/png;base64,{logo}" width="64" height="64" alt="logo" '
            'style="border-radius:14px;">'
            f'<div><div style="font-size:1.6em;font-weight:700;">{HEADER_TITLE}</div>'
            f'<div style="opacity:.85;">{HEADER_TEXT}</div></div></div>')
    except Exception:
        gr.Markdown(f"# {HEADER_TITLE}\n{HEADER_TEXT}")


def build_demo():
    # Gradio 5/6 のどちらでも動くよう、gr.Blocks() には引数を渡さない
    with gr.Blocks() as demo:
        _build_header()
        state = gr.State(None)

        with gr.Row():
            # ---- 左: 学習条件（すべてのタブで共通） ----
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 学習条件")
                board_set = gr.Radio(BOARD_SET_CHOICES, value="reachable", label="盤面の範囲")
                input_repr = gr.Radio(INPUT_REPR_CHOICES, value="onehot", label="入力表現")
                epochs = gr.Slider(1, 300, value=50, step=1, label="エポック数（学習回数）")
                n_train = gr.Slider(10, 15000, value=3000, step=10, label="学習データ数",
                                    info="学習に使える上限を超えると自動で上限に調整されます")
                hidden_layers = gr.Slider(0, 5, value=2, step=1, label="隠れ層の数",
                                          info="0にすると隠れ層のない線形モデルになります")
                units = gr.Slider(2, 256, value=64, step=2, label="隠れ層のユニット数")
                activation = gr.Dropdown(ACTIVATION_CHOICES, value="relu", label="活性化関数")
                lr = gr.Dropdown(LR_CHOICES, value=1e-3, label="学習率")
                batch_size = gr.Dropdown(BATCH_CHOICES, value=64, label="バッチサイズ")
                n_runs = gr.Slider(1, 10, value=5, step=1, label="繰り返し回数",
                                   info="同じ条件で初期値を変えて何回学習するか（平均とばらつきを表示）")
                seed = gr.Number(value=0, precision=0, label="乱数シード（1回目）")

            shared = [board_set, input_repr, epochs, n_train, hidden_layers, units,
                      activation, lr, batch_size, n_runs, seed]

            # ---- 右: タブ ----
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("① 学習実験"):
                        run_btn = gr.Button("学習を実行", variant="primary")
                        summary = gr.Markdown()
                        plot_hist = gr.Plot(label="エポックごとの正答率と損失")
                        with gr.Row():
                            plot_cm = gr.Plot(label="混同行列（テストデータ）")
                            with gr.Column():
                                per_class = gr.Dataframe(label="クラス別の正答率（テストデータ）",
                                                         interactive=False)
                                csv_file = gr.File(label="エポックごとの履歴（CSV）",
                                                   interactive=False)
                        run_btn.click(
                            on_run_experiment, inputs=shared,
                            outputs=[summary, plot_hist, plot_cm, per_class, csv_file, state])

                    with gr.Tab("② 学習データ数と正答率"):
                        gr.Markdown("左の条件のうち「学習データ数」以外を使い、"
                                    "学習データ数を変えながら正答率を比べます。")
                        sizes = gr.Textbox(value=DEFAULT_SIZES, label="学習データ数の候補（カンマ区切り）")
                        lc_btn = gr.Button("学習曲線を計算", variant="primary")
                        lc_summary = gr.Markdown()
                        lc_plot = gr.Plot(label="学習データ数と正答率")
                        lc_table = gr.Dataframe(label="数値（平均 ± 標準偏差）", interactive=False)
                        lc_csv = gr.File(label="学習曲線の数値（CSV）", interactive=False)
                        lc_btn.click(
                            on_run_learning_curve, inputs=shared + [sizes],
                            outputs=[lc_summary, lc_plot, lc_table, lc_csv])

                    with gr.Tab("③ 盤面を判定"):
                        gr.Markdown("盤面を入力すると、ルール上の正解と、"
                                    "「① 学習実験」で学習したNN（1回目のモデル）の答えを並べて表示します。\n"
                                    "左上から右下へ、1行目・2行目・3行目の順です。")
                        cells = []
                        for r in range(3):
                            with gr.Row():
                                for c in range(3):
                                    cells.append(gr.Dropdown(
                                        CELL_CHOICES, value=0, label=f"{r + 1}行{c + 1}列"))
                        with gr.Row():
                            judge_btn = gr.Button("判定", variant="primary")
                            random_btn = gr.Button("テストデータからランダムに選ぶ（学習に未使用の盤面）")
                            clear_btn = gr.Button("クリア")
                        judge_text = gr.Markdown()
                        judge_probs = gr.Label(label="NNが出した確率", num_top_classes=3)
                        judge_btn.click(on_judge, inputs=cells + [state],
                                        outputs=[judge_text, judge_probs])
                        random_btn.click(on_random_board, inputs=[board_set], outputs=cells)
                        clear_btn.click(on_clear_board, inputs=None, outputs=cells)
    return demo


def launch(share=None, debug=False, **kwargs):
    """画面を起動する。

    share=None のときは Gradio の既定に従う。ローカルでは公開リンクなし。
    Google Colab では、Gradio が自動で公開リンク（*.gradio.live）を作ることがある。
    """
    demo = build_demo()
    if FAVICON_PATH.exists() and "favicon_path" not in kwargs:
        try:
            return demo.launch(share=share, debug=debug, favicon_path=str(FAVICON_PATH), **kwargs)
        except TypeError:
            pass  # Gradio の版によって引数が無い場合は、ファビコンなしで起動する
    return demo.launch(share=share, debug=debug, **kwargs)


if __name__ == "__main__":
    launch()
