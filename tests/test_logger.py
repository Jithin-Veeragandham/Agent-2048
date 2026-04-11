"""
test_logger.py
==============

Unit tests for framework/logger.py — RunLogger and log reader functions.

All file I/O is isolated to pytest's tmp_path fixture so tests never
touch the real logs/ directory.
"""

import json
import os
import numpy as np
import pytest

from game import Action
from framework.evaluation import RewardFunction
from framework.logger import (
    RunLogger,
    load_all_runs,
    load_latest_run,
    summarize_games,
)


# ─── Helpers ──────────────────────────────────────────────────────

DUMMY_BOARD = np.array([
    [512, 256, 128,  64],
    [ 32,  16,   8,   4],
    [  2,   4,   8,  16],
    [ 32,  64, 128, 256],
], dtype=np.int32)

DUMMY_CONFIG = {"grid_size": 4, "tile_2_probability": 0.9}
DUMMY_WEIGHTS = {"tile": 1.0, "empty": 0.5, "mono": 2.5,
                 "corner": 1.5, "merge": 0.5, "smooth": 0.1}


def _make_logger(tmp_path, log_move_detail=False):
    return RunLogger(log_dir=str(tmp_path), log_move_detail=log_move_detail)


def _run_one_game(logger, agent_type="test_agent", score=1024, reached_2048=False):
    """Run the logger through one complete game cycle."""
    rf = RewardFunction()
    breakdown = rf.compute_breakdown(DUMMY_BOARD)

    logger.on_run_start(
        num_games=1,
        agent_name="TestAgent",
        agent_type=agent_type,
        agent_params={"beam_width": 10},
        config=DUMMY_CONFIG,
        reward_weights=DUMMY_WEIGHTS,
    )
    logger.on_episode_start()
    logger.log_move(
        step=0,
        state=DUMMY_BOARD,
        action=Action.LEFT,
        reward=8,
        score=score,
        inference_ms=2.5,
        reward_breakdown=breakdown,
    )
    logger.end_episode(
        final_score=score,
        highest_tile=512,
        move_count=1,
        reached_2048=reached_2048,
        final_board=DUMMY_BOARD,
        final_reward_breakdown=breakdown,
    )


# ─── File creation ────────────────────────────────────────────────

def test_logger_creates_jsonl_file(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger)
    log_path = tmp_path / "test_agent_runs.jsonl"
    assert log_path.exists()


def test_latest_run_json_created_after_save(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger)
    logger.save()
    assert (tmp_path / "latest_run.json").exists()


# ─── Record contents ──────────────────────────────────────────────

def test_run_record_has_required_fields(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger)

    log_path = tmp_path / "test_agent_runs.jsonl"
    with open(log_path) as f:
        record = json.loads(f.readline())

    for field in ('run_id', 'agent', 'agent_params', 'score',
                  'highest_tile', 'moves', 'won',
                  'final_reward_breakdown', 'avg_reward_breakdown'):
        assert field in record, f"Missing field: {field}"


def test_run_record_score_matches(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, score=888)

    log_path = tmp_path / "test_agent_runs.jsonl"
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record['score'] == 888


def test_run_record_won_false_by_default(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, reached_2048=False)

    log_path = tmp_path / "test_agent_runs.jsonl"
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record['won'] is False


def test_run_record_won_true_when_reached_2048(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, reached_2048=True)

    log_path = tmp_path / "test_agent_runs.jsonl"
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record['won'] is True


# ─── Multiple runs / appending ────────────────────────────────────

def test_logger_appends_multiple_games(tmp_path):
    logger = _make_logger(tmp_path)
    # Two separate run() calls → two lines
    _run_one_game(logger, agent_type="test_agent", score=100)
    _run_one_game(logger, agent_type="test_agent", score=200)

    log_path = tmp_path / "test_agent_runs.jsonl"
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


# ─── Log readers ──────────────────────────────────────────────────

def test_load_all_runs_returns_list(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, agent_type="beam_search")
    _run_one_game(logger, agent_type="beam_search")

    runs = load_all_runs("beam_search", log_dir=str(tmp_path))
    assert isinstance(runs, list)
    assert len(runs) == 2


def test_load_all_runs_empty_when_no_file(tmp_path):
    runs = load_all_runs("nonexistent_agent", log_dir=str(tmp_path))
    assert runs == []


def test_load_latest_run_returns_dict(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger)
    logger.save()

    result = load_latest_run(log_dir=str(tmp_path))
    assert isinstance(result, dict)
    assert 'run_id' in result


def test_load_latest_run_none_when_no_file(tmp_path):
    result = load_latest_run(log_dir=str(tmp_path))
    assert result is None


# ─── summarize_games ─────────────────────────────────────────────

def test_summarize_games_has_expected_keys(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, agent_type="test_summary", score=500)
    _run_one_game(logger, agent_type="test_summary", score=1000)

    runs = load_all_runs("test_summary", log_dir=str(tmp_path))
    summary = summarize_games(runs)

    for key in ('num_games', 'avg_score', 'max_score', 'win_rate_2048',
                 'avg_moves', 'tile_distribution'):
        assert key in summary, f"Missing key: {key}"


def test_summarize_games_avg_score(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, agent_type="avg_test", score=100)
    _run_one_game(logger, agent_type="avg_test", score=300)

    runs = load_all_runs("avg_test", log_dir=str(tmp_path))
    summary = summarize_games(runs)
    assert summary['avg_score'] == pytest.approx(200.0)


def test_summarize_games_empty_returns_empty_dict():
    assert summarize_games([]) == {}


def test_summarize_games_win_rate_zero_no_wins(tmp_path):
    logger = _make_logger(tmp_path)
    _run_one_game(logger, agent_type="wr_test", reached_2048=False)

    runs = load_all_runs("wr_test", log_dir=str(tmp_path))
    summary = summarize_games(runs)
    assert summary['win_rate_2048'] == pytest.approx(0.0)
