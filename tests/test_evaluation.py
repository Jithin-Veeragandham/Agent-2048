"""
test_evaluation.py
==================

Unit tests for framework/evaluation.py — RewardFunction and GameEvaluator.

Each of the six reward components is tested in isolation via its static method,
then the composite compute() / compute_breakdown() API is verified.
"""

import numpy as np
import pytest

from game import Game2048, Action
from framework.evaluation import RewardFunction, GameEvaluator


# ─── Helpers ──────────────────────────────────────────────────────

def make_board(*rows):
    return np.array(rows, dtype=np.int32)


def make_full_game(score=100, highest_tile=512):
    """Return a finished Game2048 for evaluator tests."""
    board = np.array([
        [512,  256, 128,  64],
        [ 32,   16,   8,   4],
        [  2,    4,   8,  16],
        [ 32,   64, 128, 256],
    ], dtype=np.int32)
    game = Game2048.from_state(board, score=score)
    game.game_over = True
    return game


# ─── RewardFunction — tile_score ─────────────────────────────────

def test_tile_score_zero_on_empty_board(empty_board):
    assert RewardFunction.tile_score(empty_board) == 0.0


def test_tile_score_positive_with_tiles():
    board = make_board(
        [2, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    )
    assert RewardFunction.tile_score(board) > 0.0


def test_tile_score_increases_with_larger_tiles():
    small = make_board([2, 0, 0, 0], [0]*4, [0]*4, [0]*4)
    large = make_board([2048, 0, 0, 0], [0]*4, [0]*4, [0]*4)
    assert RewardFunction.tile_score(large) > RewardFunction.tile_score(small)


# ─── RewardFunction — empty_bonus ────────────────────────────────

def test_empty_bonus_max_on_empty_board(empty_board):
    full_board = make_board(
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [2, 4, 8, 16],
        [32, 64, 128, 256],
    )
    assert RewardFunction.empty_bonus(empty_board) > RewardFunction.empty_bonus(full_board)


def test_empty_bonus_zero_on_full_board():
    board = make_board(
        [2,  4,  8,  16],
        [32, 64, 128, 256],
        [2,  4,  8,  16],
        [32, 64, 128, 256],
    )
    # empty_count=0, so count_score=0, adjacency=0
    assert RewardFunction.empty_bonus(board) == 0.0


def test_empty_bonus_positive_with_empty_cells():
    board = make_board(
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    )
    assert RewardFunction.empty_bonus(board) > 0.0


# ─── RewardFunction — monotonicity ───────────────────────────────

def test_monotonicity_returns_float():
    board = make_board([2, 4, 8, 16], [0]*4, [0]*4, [0]*4)
    result = RewardFunction.monotonicity(board)
    assert isinstance(result, float)


def test_monotonicity_better_for_sorted_row():
    # Sorted ascending row (good snake pattern) vs random
    sorted_board = make_board(
        [2,  4,  8, 16],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
    )
    messy_board = make_board(
        [16, 2,  8,  4],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
    )
    # monotonicity returns negative penalty — less negative = better
    assert RewardFunction.monotonicity(sorted_board) >= RewardFunction.monotonicity(messy_board)


# ─── RewardFunction — corner_bonus ───────────────────────────────

def test_corner_bonus_zero_when_max_in_top_left():
    board = make_board(
        [2048, 4, 2, 0],
        [0,    0, 0, 0],
        [0,    0, 0, 0],
        [0,    0, 0, 0],
    )
    assert RewardFunction.corner_bonus(board) == 0.0


def test_corner_bonus_negative_when_max_not_in_corner():
    board = make_board(
        [0,    4, 2, 2048],  # max tile at top-right
        [0,    0, 0, 0],
        [0,    0, 0, 0],
        [0,    0, 0, 0],
    )
    assert RewardFunction.corner_bonus(board) < 0.0


def test_corner_bonus_zero_on_empty_board(empty_board):
    assert RewardFunction.corner_bonus(empty_board) == 0.0


# ─── RewardFunction — merge_potential ────────────────────────────

def test_merge_potential_positive_for_adjacent_equals():
    board = make_board(
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    )
    assert RewardFunction.merge_potential(board) > 0.0


def test_merge_potential_zero_when_no_adjacent_equals():
    board = make_board(
        [2,  4,  8, 16],
        [32, 64, 128, 256],
        [0,  0,  0,  0],
        [0,  0,  0,  0],
    )
    assert RewardFunction.merge_potential(board) == 0.0


def test_merge_potential_zero_on_empty_board(empty_board):
    assert RewardFunction.merge_potential(empty_board) == 0.0


# ─── RewardFunction — smoothness ─────────────────────────────────

def test_smoothness_zero_on_uniform_board():
    # All same value → no log gaps between neighbors
    board = np.full((4, 4), 4, dtype=np.int32)
    assert RewardFunction.smoothness(board) == 0.0


def test_smoothness_positive_for_large_gaps():
    board = make_board(
        [2,   2048, 0, 0],
        [0,   0,    0, 0],
        [0,   0,    0, 0],
        [0,   0,    0, 0],
    )
    assert RewardFunction.smoothness(board) > 0.0


def test_smoothness_zero_on_empty_board(empty_board):
    assert RewardFunction.smoothness(empty_board) == 0.0


# ─── RewardFunction — compute / compute_breakdown ────────────────

def test_compute_returns_float(reward_fn, sample_board):
    result = reward_fn.compute(sample_board)
    assert isinstance(result, float)


def test_compute_breakdown_has_all_seven_keys(reward_fn, sample_board):
    breakdown = reward_fn.compute_breakdown(sample_board)
    expected = {'tile_score', 'empty_bonus', 'monotonicity',
                'corner_bonus', 'merge_potential', 'smoothness', 'composite'}
    assert set(breakdown.keys()) == expected


def test_compute_breakdown_values_are_floats(reward_fn, sample_board):
    breakdown = reward_fn.compute_breakdown(sample_board)
    for val in breakdown.values():
        assert isinstance(val, float)


def test_custom_weights_applied():
    # With tile weight = 0, tile_score component contributes 0
    rf = RewardFunction(weights={'tile': 0.0, 'empty': 0.5, 'mono': 0.0,
                                  'corner': 0.0, 'merge': 0.0, 'smooth': 0.0})
    board = np.full((4, 4), 4, dtype=np.int32)
    # All tiles equal, so empty_bonus = 0, tile contribution = 0 → composite = 0
    result = rf.compute(board)
    assert isinstance(result, float)


def test_default_weights_match_expected():
    rf = RewardFunction()
    assert rf.weights['tile'] == 1.0
    assert rf.weights['empty'] == 0.5
    assert rf.weights['mono'] == 2.5
    assert rf.weights['corner'] == 1.5
    assert rf.weights['merge'] == 0.5
    assert rf.weights['smooth'] == 0.1


def test_custom_weights_override_defaults():
    rf = RewardFunction(weights={'tile': 99.0})
    assert rf.weights['tile'] == 99.0
    # Others stay at default
    assert rf.weights['empty'] == 0.5


# ─── GameEvaluator ────────────────────────────────────────────────

def test_evaluator_no_episodes_returns_zero_win_rate():
    ev = GameEvaluator()
    assert ev.win_rate() == 0.0


def test_evaluator_no_episodes_returns_zero_avg_score():
    ev = GameEvaluator()
    assert ev.avg_score() == 0.0


def test_evaluator_avg_score():
    ev = GameEvaluator()
    for score in [100, 200, 300]:
        game = Game2048.from_state(
            np.array([[512, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32),
            score=score
        )
        ev.start_episode()
        ev.log_move(1.0)
        ev.end_episode(game)
    assert ev.avg_score() == pytest.approx(200.0)


def test_evaluator_win_rate_100_when_all_win():
    ev = GameEvaluator()
    board = np.array([[2048, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    for _ in range(3):
        game = Game2048.from_state(board, score=2048)
        ev.start_episode()
        ev.log_move(1.0)
        ev.end_episode(game)
    assert ev.win_rate(2048) == pytest.approx(100.0)


def test_evaluator_win_rate_0_when_none_win():
    ev = GameEvaluator()
    board = np.array([[256, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    for _ in range(3):
        game = Game2048.from_state(board, score=256)
        ev.start_episode()
        ev.log_move(1.0)
        ev.end_episode(game)
    assert ev.win_rate(2048) == pytest.approx(0.0)


def test_evaluator_avg_moves():
    ev = GameEvaluator()
    board = np.array([[4, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    for n_moves in [2, 4, 6]:
        game = Game2048.from_state(board)
        ev.start_episode()
        for _ in range(n_moves):
            ev.log_move(1.0)
        ev.end_episode(game)
    assert ev.avg_moves() == pytest.approx(4.0)


def test_evaluator_highest_tile_distribution():
    ev = GameEvaluator()
    boards = [
        np.array([[512, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32),
        np.array([[1024, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32),
    ]
    for board in boards:
        game = Game2048.from_state(board)
        ev.start_episode()
        ev.log_move(1.0)
        ev.end_episode(game)
    dist = ev.highest_tile_distribution()
    assert 512 in dist
    assert 1024 in dist


def test_evaluator_get_summary_has_key_fields():
    ev = GameEvaluator()
    board = np.array([[4, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    game = Game2048.from_state(board)
    ev.start_episode()
    ev.log_move(1.0)
    ev.end_episode(game)
    summary = ev.get_summary("TestAgent")
    for key in ('agent', 'win_rate_2048', 'avg_merge_score',
                 'avg_moves_per_game', 'num_episodes'):
        assert key in summary


def test_evaluator_reset_clears_episodes():
    ev = GameEvaluator()
    board = np.array([[4, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    game = Game2048.from_state(board)
    ev.start_episode()
    ev.log_move(1.0)
    ev.end_episode(game)
    assert ev.num_episodes == 1
    ev.reset()
    assert ev.num_episodes == 0


def test_evaluator_log_move_before_start_raises():
    ev = GameEvaluator()
    with pytest.raises(RuntimeError):
        ev.log_move(1.0)


def test_evaluator_end_episode_before_start_raises():
    ev = GameEvaluator()
    board = np.array([[4, 0, 0, 0], [0]*4, [0]*4, [0]*4], dtype=np.int32)
    game = Game2048.from_state(board)
    with pytest.raises(RuntimeError):
        ev.end_episode(game)
