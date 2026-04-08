"""
test_game_engine.py
===================

Unit tests for game/engine.py — Game2048 core logic.

All board states are constructed deterministically via Game2048.from_state()
so tests don't depend on random tile spawning.
"""

import numpy as np
import pytest

from game import Game2048, Action


# ─── Helpers ──────────────────────────────────────────────────────

def make_game(board_rows, score=0):
    """Convenience: build a Game2048 from a list of lists."""
    board = np.array(board_rows, dtype=np.int32)
    return Game2048.from_state(board, score=score)


# ─── Initialization ───────────────────────────────────────────────

def test_initial_board_has_exactly_two_tiles(config):
    game = Game2048(config)
    assert np.count_nonzero(game.get_state()) == 2


def test_initial_tiles_are_powers_of_two(config):
    game = Game2048(config)
    non_zero = game.get_state()[game.get_state() > 0]
    for val in non_zero:
        assert val in (2, 4)


def test_get_score_initially_zero(config):
    game = Game2048(config)
    assert game.get_score() == 0


# ─── State access ─────────────────────────────────────────────────

def test_get_state_returns_copy_not_reference(game):
    state = game.get_state()
    state[0, 0] = 9999
    assert game.board[0, 0] != 9999


# ─── from_state / clone ───────────────────────────────────────────

def test_from_state_preserves_board(sample_board):
    game = Game2048.from_state(sample_board, score=0)
    assert np.array_equal(game.get_state(), sample_board)


def test_from_state_preserves_score(sample_board):
    game = Game2048.from_state(sample_board, score=256)
    assert game.get_score() == 256


def test_clone_is_independent(sample_board):
    original = Game2048.from_state(sample_board)
    clone = original.clone()
    # Move clone left — should not affect original
    clone.move(Action.LEFT)
    assert np.array_equal(original.get_state(), sample_board)


def test_clone_starts_with_same_board(sample_board):
    original = Game2048.from_state(sample_board, score=100)
    clone = original.clone()
    assert np.array_equal(clone.get_state(), original.get_state())
    assert clone.get_score() == original.get_score()


# ─── Reset ────────────────────────────────────────────────────────

def test_reset_clears_score(config):
    game = Game2048(config)
    game.score = 500
    game.reset()
    assert game.get_score() == 0


def test_reset_places_initial_tiles(config):
    game = Game2048(config)
    game.reset()
    assert np.count_nonzero(game.get_state()) == 2


def test_reset_clears_game_over_flag():
    full_board = np.array([
        [2,  4,  2,  4],
        [4,  2,  4,  2],
        [2,  4,  2,  4],
        [4,  2,  4,  2],
    ], dtype=np.int32)
    game = Game2048.from_state(full_board)
    assert game.is_game_over()
    game.reset()
    assert not game.game_over


# ─── Move mechanics — LEFT ────────────────────────────────────────

def test_move_left_merges_row():
    game = make_game([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    valid, reward = game.move(Action.LEFT)
    assert valid
    assert reward == 4
    assert game.get_score() == 4
    # Merged tile must be at row 0, col 0
    assert game.board[0, 0] == 4


def test_move_left_slides_tiles():
    game = make_game([
        [0, 0, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    game.move(Action.LEFT)
    assert game.board[0, 0] == 4


def test_no_double_merge_in_one_move():
    # [2,2,2,2] -> [4,4,0,0], NOT [8,0,0,0]
    game = make_game([
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    game.move(Action.LEFT)
    assert game.board[0, 0] == 4
    assert game.board[0, 1] == 4


def test_merge_increments_score():
    game = make_game([
        [4, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    _, reward = game.move(Action.LEFT)
    assert reward == 8
    assert game.get_score() == 8


# ─── Move mechanics — RIGHT ───────────────────────────────────────

def test_move_right_merges_row():
    game = make_game([
        [0, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    valid, reward = game.move(Action.RIGHT)
    assert valid
    assert reward == 4
    assert game.board[0, 3] == 4


# ─── Move mechanics — UP ─────────────────────────────────────────

def test_move_up_merges_column():
    game = make_game([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    valid, reward = game.move(Action.UP)
    assert valid
    assert reward == 4
    assert game.board[0, 0] == 4


# ─── Move mechanics — DOWN ───────────────────────────────────────

def test_move_down_merges_column():
    game = make_game([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [2, 0, 0, 0],
    ])
    valid, reward = game.move(Action.DOWN)
    assert valid
    assert reward == 4
    assert game.board[3, 0] == 4


# ─── Invalid move ─────────────────────────────────────────────────

def test_invalid_move_returns_false():
    # All tiles on left column — LEFT does nothing
    game = make_game([
        [2,  0, 0, 0],
        [4,  0, 0, 0],
        [8,  0, 0, 0],
        [16, 0, 0, 0],
    ])
    valid, reward = game.move(Action.LEFT)
    assert not valid
    assert reward == 0


def test_invalid_move_does_not_change_board():
    board = np.array([
        [2,  0, 0, 0],
        [4,  0, 0, 0],
        [8,  0, 0, 0],
        [16, 0, 0, 0],
    ], dtype=np.int32)
    game = Game2048.from_state(board)
    game.move(Action.LEFT)
    assert np.array_equal(game.get_state(), board)


def test_invalid_move_does_not_spawn_tile():
    game = make_game([
        [2,  0, 0, 0],
        [4,  0, 0, 0],
        [8,  0, 0, 0],
        [16, 0, 0, 0],
    ])
    before = np.count_nonzero(game.get_state())
    game.move(Action.LEFT)
    after = np.count_nonzero(game.get_state())
    assert before == after


# ─── Tile spawning ────────────────────────────────────────────────

def test_valid_move_spawns_new_tile():
    game = make_game([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    before = np.count_nonzero(game.get_state())
    game.move(Action.LEFT)
    after = np.count_nonzero(game.get_state())
    # Two 2s merged into one 4, so before=2, after=2 (one merged tile + one new)
    assert after == before


# ─── move_fast ────────────────────────────────────────────────────

def test_move_fast_returns_true_for_valid_move():
    game = make_game([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert game.move_fast(Action.LEFT) is True


def test_move_fast_returns_false_for_invalid_move():
    game = make_game([
        [2,  0, 0, 0],
        [4,  0, 0, 0],
        [8,  0, 0, 0],
        [16, 0, 0, 0],
    ])
    assert game.move_fast(Action.LEFT) is False


# ─── Game over detection ──────────────────────────────────────────

def test_is_game_over_false_on_fresh_game(config):
    game = Game2048(config)
    assert not game.is_game_over()


def test_is_game_over_false_when_empty_cells_exist():
    game = make_game([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 0],   # one empty cell
        [4, 2, 4, 2],
    ])
    assert not game.is_game_over()


def test_is_game_over_false_when_merges_available():
    # Full board but row 3 has two adjacent 4s
    game = make_game([
        [2,  4,  2,  4],
        [4,  2,  4,  2],
        [2,  4,  2,  4],
        [4,  2,  4,  4],   # mergeable pair at end
    ])
    assert not game.is_game_over()


def test_is_game_over_true_when_board_full_no_merges():
    # Checkerboard — no adjacent equal tiles, completely full
    game = make_game([
        [2,  4,  2,  4],
        [4,  2,  4,  2],
        [2,  4,  2,  4],
        [4,  2,  4,  2],
    ])
    assert game.is_game_over()


# ─── Available moves ─────────────────────────────────────────────

def test_available_moves_nonempty_on_fresh_game(config):
    game = Game2048(config)
    assert len(game.get_available_moves()) >= 1


def test_available_moves_empty_on_game_over():
    game = make_game([
        [2,  4,  2,  4],
        [4,  2,  4,  2],
        [2,  4,  2,  4],
        [4,  2,  4,  2],
    ])
    assert game.get_available_moves() == []


def test_available_moves_excludes_noop_direction():
    # Tiles in left column only — LEFT is a no-op
    game = make_game([
        [2,  0, 0, 0],
        [4,  0, 0, 0],
        [8,  0, 0, 0],
        [16, 0, 0, 0],
    ])
    assert Action.LEFT not in game.get_available_moves()


def test_available_moves_returns_list_of_actions(config):
    game = Game2048(config)
    moves = game.get_available_moves()
    for move in moves:
        assert isinstance(move, Action)


# ─── Action enum ─────────────────────────────────────────────────

def test_action_enum_values():
    assert Action.UP.value    == 0
    assert Action.DOWN.value  == 1
    assert Action.LEFT.value  == 2
    assert Action.RIGHT.value == 3
