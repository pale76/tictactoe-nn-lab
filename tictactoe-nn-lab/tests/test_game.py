"""ルール部分のテスト。ここが間違っていると実験全体が無意味になるので、特に丁寧に確認する。"""
import pytest

from tictactoe_nn.game import (WIN_LINES, generate_all_boards, generate_reachable_boards,
                               has_line, is_reachable, label_of, next_player, str_to_board,
                               board_to_str)


def test_reachable_board_count():
    assert len(generate_reachable_boards()) == 5478


def test_all_board_count():
    assert len(generate_all_boards()) == 3 ** 9


def test_reachable_is_subset_of_all():
    assert set(generate_reachable_boards()) <= set(generate_all_boards())


def test_finished_position_counts_match_known_values():
    """プレイヤー1先手の〇×ゲームの終局盤面は、P1勝ち626・P2勝ち316・引き分け16の計958通り。"""
    boards = generate_reachable_boards()
    p1 = sum(1 for b in boards if label_of(b) == 1)
    p2 = sum(1 for b in boards if label_of(b) == 2)
    assert p1 == 626
    assert p2 == 316
    draws = sum(1 for b in boards if label_of(b) == 0 and 0 not in b)
    assert draws == 16
    # ラベル0（未決着 + 引き分け）は残り全部
    assert sum(1 for b in boards if label_of(b) == 0) == 5478 - 626 - 316


def test_every_win_line_is_detected_for_both_players():
    for player in (1, 2):
        for line in WIN_LINES:
            board = [0] * 9
            for i in line:
                board[i] = player
            assert has_line(tuple(board), player)
            assert label_of(tuple(board)) == player


def test_label_examples():
    assert label_of(str_to_board("000000000")) == 0          # 空の盤面
    assert label_of(str_to_board("111220000")) == 1          # P1が上段を揃えた
    assert label_of(str_to_board("122121100")) == 1          # P1が左列を揃えた
    assert label_of(str_to_board("221112212")) == 0          # 引き分け（勝者なし）
    assert label_of(str_to_board("012210010")) == 1          # 中央の縦列(1,4,7)がすべて1
    assert label_of(str_to_board("221011000")) == 0          # 未決着


def test_no_reachable_board_has_two_winners():
    for b in generate_reachable_boards():
        assert not (has_line(b, 1) and has_line(b, 2))


def test_reachable_boards_have_valid_piece_counts():
    for b in generate_reachable_boards():
        assert b.count(1) - b.count(2) in (0, 1)


def test_next_player():
    assert next_player(str_to_board("000000000")) == 1
    assert next_player(str_to_board("100000000")) == 2
    assert next_player(str_to_board("120000000")) == 1


def test_is_reachable():
    assert is_reachable(str_to_board("000000000"))
    assert not is_reachable(str_to_board("111111111"))   # 1が9個は現れない
    assert not is_reachable(str_to_board("222000000"))   # 2だけが多い


def test_str_roundtrip_and_validation():
    b = str_to_board("012210010")
    assert board_to_str(b) == "012210010"
    with pytest.raises(ValueError):
        str_to_board("01221001")      # 8文字
    with pytest.raises(ValueError):
        str_to_board("012210013")     # 3が混ざっている
