"""
test_interaction.py
===================

Unit tests for framework/interaction.py — InteractionModule.

Uses a MinimalAgent (always picks the first available move) to keep
tests fast and deterministic, without any real search.
"""

import numpy as np
import pytest

from game import Game2048, Action
from agents.base import BaseAgent
from framework.evaluation import RewardFunction
from framework.interaction import InteractionModule


# ─── Minimal test agent ───────────────────────────────────────────

class MinimalAgent(BaseAgent):
    """Always picks the first available action. No search, instant."""

    agent_type = "minimal"

    def __init__(self):
        super().__init__("MinimalAgent")

    def choose_action(self, state, available_moves, game_context=None):
        return available_moves[0]

    def get_params(self):
        return {}


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return MinimalAgent()


@pytest.fixture
def module(config, agent):
    return InteractionModule(config=config, agent=agent, verbose=False)


# ─── run_episode ─────────────────────────────────────────────────

def test_run_episode_returns_dict(module):
    result = module.run_episode()
    assert isinstance(result, dict)


def test_run_episode_has_score_key(module):
    result = module.run_episode()
    assert 'score' in result


def test_run_episode_score_is_nonnegative(module):
    result = module.run_episode()
    assert result['score'] >= 0


def test_run_episode_has_highest_tile(module):
    result = module.run_episode()
    assert 'highest_tile' in result
    assert result['highest_tile'] >= 2


def test_run_episode_has_moves(module):
    result = module.run_episode()
    assert 'moves' in result
    assert result['moves'] >= 1


def test_run_episode_has_reached_2048_key(module):
    result = module.run_episode()
    assert 'reached_2048' in result
    assert isinstance(result['reached_2048'], bool)


def test_run_episode_has_final_board(module):
    result = module.run_episode()
    assert 'final_board' in result
    assert isinstance(result['final_board'], np.ndarray)
    assert result['final_board'].shape == (4, 4)


# ─── run (multi-game) ─────────────────────────────────────────────

def test_run_returns_list(module):
    results = module.run(num_games=2)
    assert isinstance(results, list)


def test_run_returns_correct_count(module):
    results = module.run(num_games=3)
    assert len(results) == 3


def test_run_each_result_has_score(module):
    results = module.run(num_games=2)
    for r in results:
        assert 'score' in r
        assert r['score'] >= 0


# ─── get_results / reset ─────────────────────────────────────────

def test_get_results_after_run_has_avg_score(module):
    module.run(num_games=2)
    summary = module.get_results()
    assert 'avg_merge_score' in summary


def test_get_results_num_episodes_matches(module):
    module.run(num_games=2)
    summary = module.get_results()
    assert summary['num_episodes'] == 2


def test_reset_clears_episode_count(module):
    module.run(num_games=2)
    module.reset()
    summary = module.get_results()
    assert summary['num_episodes'] == 0


# ─── Custom reward function ──────────────────────────────────────

def test_run_with_custom_reward_fn(config, agent):
    custom_rf = RewardFunction(weights={'tile': 2.0, 'empty': 1.0, 'mono': 0.0,
                                         'corner': 0.0, 'merge': 0.0, 'smooth': 0.0})
    module = InteractionModule(config=config, agent=agent,
                                reward_fn=custom_rf, verbose=False)
    results = module.run(num_games=1)
    assert len(results) == 1
    assert results[0]['score'] >= 0
