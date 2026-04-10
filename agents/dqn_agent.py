"""
agents/dqn_agent.py
===================

Two DQN agents wrapped as BaseAgent subclasses, plugging directly
into the Agent-2048 InteractionModule and RunLogger.

    ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
    Γöé Agent           Γöé Details                                      Γöé
    Γö£ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöñ
    Γöé DQNAgent4x4     Γöé achraf-azize architecture                    Γöé
    Γöé                 Γöé Parallel 1├ù2 / 2├ù1 convolutions, 2 levels   Γöé
    Γöé                 Γöé Trained with nb_empty_tiles reward           Γöé
    Γöé                 Γöé Checkpoint key: 'online'                     Γöé
    Γöé                 Γöé Trained: 3 000 epochs (~3 hrs GPU)           Γöé
    Γöé                 Γöé 2048 rate: ~3ΓÇô5%                             Γöé
    Γö£ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöñ
    Γöé DQNAgent6x6     Γöé ResNet trunk architecture                    Γöé
    Γöé                 Γöé 4 residual blocks, 128 channels              Γöé
    Γöé                 Γöé Trained with log2(merge) reward              Γöé
    Γöé                 Γöé Checkpoint key: 'policy_net'                 Γöé
    ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö┤ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ

Usage
-----
    # 4├ù4
    from agents.dqn_agent import DQNAgent4x4
    agent = DQNAgent4x4(model_path="model_2048.pt")

    # 6├ù6
    from agents.dqn_agent import DQNAgent6x6
    agent = DQNAgent6x6(model_path="dqn_model.pt")

    # Both share the same eval pattern:
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    logger = RunLogger()
    module = InteractionModule(
        config={"grid_size": 4},   # or 6
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=100)
    module.print_results()

Quick eval from command line
----------------------------
    # 4├ù4
    python agents/dqn_agent.py --model model_2048.pt --grid 4 --games 100

    # 6├ù6
    python agents/dqn_agent.py --model dqn_model.pt  --grid 6 --games 100
"""

import math
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

# Ensure repo root is on the path regardless of where the script is invoked from
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from game import Action
from agents.base import BaseAgent


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  SHARED CONSTANTS
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

# Action order used during training in play2048.py (4├ù4 model):
#   index 0 = left, 1 = up, 2 = right, 3 = down
# Repo Action enum values:
#   UP=0, DOWN=1, LEFT=2, RIGHT=3
# Bridge: Q-network output index ΓåÆ repo Action enum
_4X4_IDX_TO_ACTION = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]
_4X4_ACTION_TO_IDX = {
    Action.LEFT:  0,
    Action.UP:    1,
    Action.RIGHT: 2,
    Action.DOWN:  3,
}

# 6├ù6 model ACTIONS = [UP, DOWN, LEFT, RIGHT] matches Action.value directly
_6X6_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

MAX_TILE_POWER = 17   # used by 6├ù6 one-hot encoder


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  4├ù4 STATE ENCODING ΓÇö achraf-azize / play2048.py
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _encode_4x4(board: np.ndarray) -> torch.Tensor:
    """(4, 4) int ΓåÆ (1, 16, 4, 4) float tensor.

    Identical to play2048.py encode() so the weights load correctly.
    """
    power_mat = np.zeros((1, 16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                power_mat[0, 0, i, j] = 1.0
            else:
                power = int(math.log(board[i, j], 2))
                power_mat[0, power, i, j] = 1.0
    return torch.from_numpy(np.flip(power_mat, axis=0).copy())


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  6├ù6 STATE ENCODING ΓÇö identical to dqn_agent.py (6├ù6 branch)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _encode_6x6(board: np.ndarray) -> np.ndarray:
    """(H, W) int ΓåÆ (MAX_TILE_POWER+1, H, W) float32 ndarray.

    Identical to encode_board() in the 6├ù6 dqn_agent.py so the
    weights load correctly.
    """
    rows, cols = board.shape
    encoded = np.zeros((MAX_TILE_POWER + 1, rows, cols), dtype=np.float32)
    encoded[0] = (board == 0).astype(np.float32)
    for r in range(rows):
        for c in range(cols):
            val = board[r, c]
            if val > 0:
                power = int(np.log2(val))
                if power <= MAX_TILE_POWER:
                    encoded[power][r][c] = 1.0
    return encoded


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  4├ù4 NETWORK ΓÇö achraf-azize architecture
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

class _DQN4x4(nn.Module):
    """Parallel 1├ù2 / 2├ù1 convolutions, two levels deep."""
    def __init__(self):
        super().__init__()
        self.conv_a  = nn.Conv2d(16,  128, kernel_size=(1, 2))
        self.conv_b  = nn.Conv2d(16,  128, kernel_size=(2, 1))
        self.conv_aa = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_ab = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.conv_ba = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_bb = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.fc = nn.Sequential(
            nn.Linear(7424, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_a  = F.relu(self.conv_a(x))
        x_b  = F.relu(self.conv_b(x))
        x_aa = F.relu(self.conv_aa(x_a))
        x_ab = F.relu(self.conv_ab(x_a))
        x_ba = F.relu(self.conv_ba(x_b))
        x_bb = F.relu(self.conv_bb(x_b))
        concat = torch.cat([
            x_a.flatten(1),  x_b.flatten(1),
            x_aa.flatten(1), x_ab.flatten(1),
            x_ba.flatten(1), x_bb.flatten(1),
        ], dim=1)
        return self.fc(concat)


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  6├ù6 NETWORK ΓÇö ResNet trunk architecture
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + res)


class _QNetwork6x6(nn.Module):
    """ResNet trunk + single Q-head. Default: 4 residual blocks, 128 ch."""
    def __init__(self, grid_size: int = 6, num_res_blocks: int = 4,
                 channels: int = 128):
        super().__init__()
        in_ch = MAX_TILE_POWER + 1
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *[_ResidualBlock(channels) for _ in range(num_res_blocks)],
        )
        self.q_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.trunk(x))


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  DQNAgent4x4 ΓÇö BaseAgent wrapper for the 4├ù4 model
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

class DQNAgent4x4(BaseAgent):
    """DQN agent using the achraf-azize architecture (4├ù4 board).

    Args:
        model_path:    Path to .pt checkpoint (must contain 'online' key).
        device:        'cuda', 'cpu', or None (auto-detect).
        deterministic: True = argmax Q (default). False = softmax sample.
    """

    agent_type: str = "dqn_4x4"

    def __init__(
        self,
        model_path: str = "model_2048.pt",
        device: Optional[str] = None,
        deterministic: bool = True,
    ):
        super().__init__("DQN-4x4 (achraf-azize)")
        self.model_path    = model_path
        self.deterministic = deterministic
        self.device        = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else
             ("mps" if hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available() else "cpu"))
        )

        self.net = _DQN4x4().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=False)
        self.net.load_state_dict(ckpt["online"])
        self.net.eval()

        self._train_epochs   = ckpt.get("epoch", "unknown")
        self._train_epsilon  = ckpt.get("epsilon", "unknown")
        self._best_tile_seen = (
            max(ckpt["max_tiles"]) if ckpt.get("max_tiles") else "unknown"
        )
        print(
            f"DQNAgent4x4: loaded '{model_path}' | "
            f"epoch={self._train_epochs} | "
            f"best_tile={self._best_tile_seen} | "
            f"device={self.device}"
        )

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        x = _encode_4x4(state).to(self.device)
        with torch.no_grad():
            q = self.net(x).squeeze(0)

        if self.deterministic:
            action = max(available_moves,
                         key=lambda a: q[_4X4_ACTION_TO_IDX[a]].item())
        else:
            legal_idx = [_4X4_ACTION_TO_IDX[a] for a in available_moves]
            legal_q   = torch.stack([q[i] for i in legal_idx])
            probs     = torch.softmax(legal_q, dim=0).cpu().numpy()
            chosen_i  = np.random.choice(len(available_moves), p=probs)
            action    = available_moves[chosen_i]
        return action

    def get_params(self) -> Dict:
        return {
            "architecture":   "achraf-azize (parallel 1├ù2 / 2├ù1 conv)",
            "grid_size":      4,
            "reward":         "nb_empty_tiles",
            "train_epochs":   self._train_epochs,
            "best_tile_seen": self._best_tile_seen,
            "deterministic":  self.deterministic,
            "device":         str(self.device),
            "model_path":     self.model_path,
        }


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  DQNAgent6x6 ΓÇö BaseAgent wrapper for the 6├ù6 model
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

class DQNAgent6x6(BaseAgent):
    """DQN agent using the ResNet trunk architecture (6├ù6 board).

    Args:
        model_path:    Path to .pt checkpoint (must contain 'policy_net' key).
        grid_size:     Board dimension (default 6).
        num_res_blocks: Residual blocks ΓÇö must match training config.
        channels:      Conv channels ΓÇö must match training config.
        device:        'cuda', 'cpu', or None (auto-detect).
    """

    agent_type: str = "dqn_6x6"

    def __init__(
        self,
        model_path: str = "dqn_model.pt",
        grid_size: int = 6,
        num_res_blocks: int = 4,
        channels: int = 128,
        device: Optional[str] = None,
    ):
        super().__init__(f"DQN-{grid_size}x{grid_size} (ResNet)")
        self.model_path = model_path
        self.grid_size  = grid_size
        self.device     = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else
             ("mps" if hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available() else "cpu"))
        )

        self.net = _QNetwork6x6(grid_size, num_res_blocks, channels).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=False)
        self.net.load_state_dict(ckpt["policy_net"])
        self.net.eval()

        self._train_timesteps = ckpt.get("total_timesteps", "unknown")
        self._train_episodes  = ckpt.get("total_episodes",  "unknown")
        print(
            f"DQNAgent6x6: loaded '{model_path}' | "
            f"steps={self._train_timesteps} | "
            f"episodes={self._train_episodes} | "
            f"device={self.device}"
        )

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        encoded = _encode_6x6(state)
        tensor  = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(tensor).squeeze(0)

        # Mask illegal actions to -inf then argmax
        mask = torch.full((4,), float('-inf'), device=self.device)
        for a in available_moves:
            mask[a.value] = q[a.value]

        action_idx = mask.argmax().item()
        chosen     = _6X6_ACTIONS[action_idx]

        if chosen not in available_moves:
            chosen = available_moves[0]
        return chosen

    def get_params(self) -> Dict:
        return {
            "architecture":    "ResNet trunk (4 residual blocks, 128ch)",
            "grid_size":       self.grid_size,
            "reward":          "log2(merge_reward)",
            "train_timesteps": self._train_timesteps,
            "train_episodes":  self._train_episodes,
            "device":          str(self.device),
            "model_path":      self.model_path,
        }


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  CLI ΓÇö run eval and write log files
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate DQN agents (4├ù4 or 6├ù6) and write log files.")
    parser.add_argument("--model",   required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--grid",    type=int, default=4, choices=[4, 6],
                        help="Board size: 4 for 4├ù4 model, 6 for 6├ù6 model")
    parser.add_argument("--games",   type=int, default=100,
                        help="Number of evaluation games")
    parser.add_argument("--config",  default="config.json",
                        help="Path to game config JSON")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory to write log files")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        config["grid_size"] = args.grid
    else:
        config = {
            "grid_size": args.grid,
            "tile_2_probability": 0.9,
            "initial_tiles": 2,
            "random_seed": None,
        }

    if args.grid == 4:
        agent            = DQNAgent4x4(model_path=args.model)
        results_filename = "dqn_4x4_eval_results.json"
    else:
        agent            = DQNAgent6x6(model_path=args.model)
        results_filename = "dqn_6x6_eval_results.json"

    os.makedirs(args.log_dir, exist_ok=True)
    logger = RunLogger(log_dir=args.log_dir)

    module = InteractionModule(
        config=config,
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=args.games)
    module.print_results()

    results_path = os.path.join(args.log_dir, results_filename)
    module.save_results(results_path)

    # ΓöÇΓöÇ Convert JSONL ΓåÆ wrapped JSON (matches PPO log format) ΓöÇΓöÇΓöÇΓöÇ
    jsonl_path   = os.path.join(args.log_dir, f"{agent.agent_type}_runs.jsonl")
    wrapped_path = os.path.join(args.log_dir, f"{agent.agent_type}_runs.json")

    games = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))

    if games:
        first = games[0]
        wrapped = {
            "run_id":         first.get("run_id"),
            "timestamp":      first.get("timestamp"),
            "agent":          first.get("agent"),
            "agent_type":     agent.agent_type,
            "agent_params":   first.get("agent_params", {}),
            "config":         first.get("config", {}),
            "reward_weights": first.get("reward_weights", {}),
            "num_games":      len(games),
            "games":          games,
        }
        with open(wrapped_path, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, indent=2)
        print(f"\nLog file  -> {jsonl_path}")
        print(f"Wrapped   -> {wrapped_path}")
        print(f"Results   -> {results_path}")
    else:
        print(f"\nLog file  -> {jsonl_path}")
        print(f"Results   -> {results_path}")
