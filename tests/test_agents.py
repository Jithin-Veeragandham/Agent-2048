"""
test_agents.py
==============

Smoke tests for all non-RL agents: verify that each agent returns a
valid Action that is in the list of available moves.

These tests are intentionally minimal — they check the agent's
public contract (does it output a valid move?) without evaluating
strategy or performance.

AlphaZero and PPO are skipped here because they require pre-trained
model files (torch checkpoints) that may not be present.
"""

import numpy as np
import pytest

from game import Game2048, Action
from framework.evaluation import REWARD_SEARCH


# ─── Shared board for all agent tests ────────────────────────────
#
# A mid-game board with several available moves in all directions.
# Using from_state() ensures determinism — no random tile spawn.

MIDGAME_BOARD = np.array([
    [ 2,  2,  4,  8],
    [ 4,  8, 16, 32],
    [ 8, 16, 32, 64],
    [16, 32, 64,  0],   # one empty cell keeps all 4 directions available
], dtype=np.int32)


def make_game_context():
    game = Game2048.from_state(MIDGAME_BOARD)
    available = game.get_available_moves()
    state = game.get_state()
    context = {
        'game': game,
        'score': game.get_score(),
        'move_number': 50,
        'reward_fn': REWARD_SEARCH,
    }
    return state, available, context


# ─── BeamSearchAgent ─────────────────────────────────────────────

def test_beam_search_returns_valid_action():
    from agents.beam_search import BeamSearchAgent
    agent = BeamSearchAgent(beam_width=3, search_depth=3)

    state, available, context = make_game_context()
    action = agent.choose_action(state, available, context)

    assert isinstance(action, Action)
    assert action in available


# ─── MCTSAgent ───────────────────────────────────────────────────

def test_mcts_returns_valid_action():
    from agents.mcts import MCTSAgent
    agent = MCTSAgent(num_simulations=10, rollout_depth=3)

    state, available, context = make_game_context()
    action = agent.choose_action(state, available, context)

    assert isinstance(action, Action)
    assert action in available


# ─── ExpectimaxAgent ─────────────────────────────────────────────

def test_expectimax_returns_valid_action():
    from agents.expectimax import ExpectimaxAgent
    agent = ExpectimaxAgent(depth=2)

    state, available, context = make_game_context()
    action = agent.choose_action(state, available, context)

    assert isinstance(action, Action)
    assert action in available


# ─── ExpectimaxSnakeAgent ─────────────────────────────────────────

def test_expectimax_snake_returns_valid_action():
    from agents.expectimax_snake import ExpectimaxSnakeAgent
    agent = ExpectimaxSnakeAgent(depth=2)

    state, available, context = make_game_context()
    action = agent.choose_action(state, available, context)

    assert isinstance(action, Action)
    assert action in available


# ─── Agent hooks — called without error ──────────────────────────

@pytest.mark.parametrize("AgentClass,kwargs", [
    ("agents.beam_search.BeamSearchAgent",   {"beam_width": 3, "search_depth": 3}),
    ("agents.mcts.MCTSAgent",                {"num_simulations": 5}),
    ("agents.expectimax.ExpectimaxAgent",    {"depth": 2}),
])
def test_agent_hooks_do_not_raise(AgentClass, kwargs):
    """on_episode_start / on_move_result / on_episode_end should not raise."""
    import importlib
    module_path, class_name = AgentClass.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    agent = cls(**kwargs)

    state, available, context = make_game_context()

    agent.on_episode_start()
    action = agent.choose_action(state, available, context)
    agent.on_move_result(state, action, 0, state, False)
    agent.on_episode_end(state, 100)


# ─── get_params returns a dict ────────────────────────────────────

@pytest.mark.parametrize("AgentClass,kwargs", [
    ("agents.beam_search.BeamSearchAgent",   {"beam_width": 5, "search_depth": 5}),
    ("agents.mcts.MCTSAgent",                {"num_simulations": 5}),
    ("agents.expectimax.ExpectimaxAgent",    {"depth": 2}),
    ("agents.expectimax_snake.ExpectimaxSnakeAgent", {"depth": 2}),
])
def test_get_params_returns_dict(AgentClass, kwargs):
    import importlib
    module_path, class_name = AgentClass.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    agent = cls(**kwargs)

    params = agent.get_params()
    assert isinstance(params, dict)
