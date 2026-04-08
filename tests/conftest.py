"""
Shared fixtures for the 2048 test suite.

All tests import from the project root, which pytest adds to sys.path
via the ``pythonpath = .`` setting in pytest.ini.
"""

import pytest
import numpy as np

from game import Game2048, Action
from framework.evaluation import RewardFunction


DEFAULT_CONFIG = {
    "grid_size": 4,
    "tile_2_probability": 0.9,
    "initial_tiles": 2,
    "random_seed": 42,
}


@pytest.fixture
def config():
    return DEFAULT_CONFIG.copy()


@pytest.fixture
def empty_board():
    return np.zeros((4, 4), dtype=np.int32)


@pytest.fixture
def sample_board():
    """4x4 board with merges available in all directions. Max tile = 16."""
    return np.array([
        [ 2,  2,  0,  0],
        [ 4,  0,  4,  0],
        [ 0,  0,  8,  8],
        [16,  0,  0, 16],
    ], dtype=np.int32)


@pytest.fixture
def game(config):
    return Game2048(config)


@pytest.fixture
def reward_fn():
    return RewardFunction()
