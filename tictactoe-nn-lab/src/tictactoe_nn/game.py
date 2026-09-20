"""盤面の生成と、ルールに基づく正解ラベルの付与。

盤面は 9 個の整数のタプルで表す（左上 → 右下の順）。
    0: 空白 / 1: プレイヤー1(〇) / 2: プレイヤー2(×)

ラベル（ニューラルネットワークが当てる正解）:
    0: 未決着または引き分け
    1: プレイヤー1の勝ち
    2: プレイヤー2の勝ち
"""
from __future__ import annotations

import itertools
from functools import lru_cache

EMPTY, P1, P2 = 0, 1, 2

# 表示用の名前
SYMBOLS = {EMPTY: "空", P1: "〇", P2: "×"}
LABEL_NAMES = {
    0: "未決着/引き分け",
    1: "プレイヤー1(〇)の勝ち",
    2: "プレイヤー2(×)の勝ち",
}

# 3つ並びになる8通りのマスの組み合わせ
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 横
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 縦
    (0, 4, 8), (2, 4, 6),             # 斜め
)

Board = tuple  # 長さ9の整数タプル


def has_line(board: Board, player: int) -> bool:
    """player が3つ並べているか。"""
    return any(all(board[i] == player for i in line) for line in WIN_LINES)


def label_of(board: Board) -> int:
    """盤面の正解ラベル(0/1/2)をルールから求める。

    両者が同時に3つ並べている盤面は、実際のゲームでは現れない。
    「全パターン」モードのために便宜上 0 とする。
    """
    win1 = has_line(board, P1)
    win2 = has_line(board, P2)
    if win1 and not win2:
        return 1
    if win2 and not win1:
        return 2
    return 0


def is_full(board: Board) -> bool:
    return all(cell != EMPTY for cell in board)


def is_finished(board: Board) -> bool:
    """勝敗が付いたか、盤面が埋まっていて、これ以上打てない状態か。"""
    return has_line(board, P1) or has_line(board, P2) or is_full(board)


def next_player(board: Board) -> int:
    """プレイヤー1が先手。1と2の数が同じなら次は1、そうでなければ2。"""
    return P1 if board.count(P1) == board.count(P2) else P2


def generate_reachable_boards() -> list:
    """プレイヤー1先手で、実際のゲームで現れる盤面をすべて生成する（5,478通り）。

    空の盤面から交互に打ち、勝敗が付いた盤面からは先へ進めない。
    """
    start = (EMPTY,) * 9
    seen = {start}
    stack = [start]
    while stack:
        board = stack.pop()
        if is_finished(board):
            continue
        player = next_player(board)
        for i in range(9):
            if board[i] == EMPTY:
                nxt = board[:i] + (player,) + board[i + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return sorted(seen)


def generate_all_boards() -> list:
    """0/1/2 の全組み合わせ（3^9 = 19,683通り）。ありえない盤面も含む。"""
    return [tuple(b) for b in itertools.product(range(3), repeat=9)]


@lru_cache(maxsize=1)
def _reachable_set() -> frozenset:
    return frozenset(generate_reachable_boards())


def is_reachable(board: Board) -> bool:
    """実際のゲーム（プレイヤー1先手）で現れうる盤面か。"""
    return tuple(board) in _reachable_set()


# ---- 文字列との相互変換（例: "012210010"） ----

def board_to_str(board: Board) -> str:
    return "".join(str(c) for c in board)


def str_to_board(text: str) -> Board:
    text = text.strip()
    if len(text) != 9 or any(ch not in "012" for ch in text):
        raise ValueError(f"盤面は 0/1/2 の9文字で指定してください: {text!r}")
    return tuple(int(ch) for ch in text)


def board_to_text(board: Board) -> str:
    """3x3の見やすい文字列にする（〇/×/・）。"""
    mark = {EMPTY: "・", P1: "〇", P2: "×"}
    rows = ["".join(mark[board[r * 3 + c]] for c in range(3)) for r in range(3)]
    return "\n".join(rows)
