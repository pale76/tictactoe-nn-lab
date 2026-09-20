"""data/sample.csv が、ゲームのルールおよびアプリのデータ分割と一致しているか確認する。"""
import csv
from pathlib import Path

from tictactoe_nn.data import build_dataset
from tictactoe_nn.game import generate_reachable_boards, label_of

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.csv"


def _read_rows():
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [({"board": tuple(int(row[f"cell_{i}"]) for i in range(1, 10)),
                  "label": int(row["label"]), "split": row["split"]}) for row in reader]


def test_sample_csv_covers_all_reachable_boards():
    rows = _read_rows()
    assert len(rows) == 5478
    assert {r["board"] for r in rows} == set(generate_reachable_boards())


def test_sample_csv_labels_match_rules():
    for r in _read_rows():
        assert r["label"] == label_of(r["board"])


def test_sample_csv_split_matches_app_split():
    ds = build_dataset("reachable")
    test_boards = {tuple(int(v) for v in b) for b in ds.test_boards}
    for r in _read_rows():
        assert (r["split"] == "test") == (r["board"] in test_boards)
