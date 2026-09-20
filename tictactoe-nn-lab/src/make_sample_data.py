"""data/sample.csv を作り直すスクリプト。

実際のゲーム（プレイヤー1先手）で現れる盤面5,478通りと、ルールから付けた正解ラベル、
学習/テストの区分（アプリと同じ分割）を書き出す。

使い方:
    python src/make_sample_data.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tictactoe_nn.data import build_dataset  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.csv"


def main():
    ds = build_dataset("reachable")
    rows = []
    for boards, labels, split in ((ds.train_boards, ds.train_labels, "train_pool"),
                                  (ds.test_boards, ds.test_labels, "test")):
        for board, label in zip(boards, labels):
            rows.append((tuple(int(v) for v in board), int(label), split))
    rows.sort()  # 盤面の順に並べる

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([f"cell_{i}" for i in range(1, 10)] + ["label", "split"])
        for board, label, split in rows:
            writer.writerow(list(board) + [label, split])
    print(f"{len(rows)}行を書き出しました: {OUT_PATH}")


if __name__ == "__main__":
    main()
