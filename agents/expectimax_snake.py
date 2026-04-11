"""
expectimax_snake.py
===================

Expectimax agent using the mschrandt/2048 snake heuristic.

Heuristic: weighted dot product of board values against PERFECT_SNAKE,
a snake-pattern matrix that encourages the largest tile to anchor in
the top-left and tiles to decrease in a zigzag order.

Search: integer/float depth trick alternating CHANCE (integer) and
MAX (float) nodes, decrementing by 0.5 each level. Only 2-tiles are
spawned (matching mschrandt/2048 ai.py). Depth=2 = 3 chance + 2 player
levels.

Usage::

    from agents.expectimax_snake import ExpectimaxSnakeAgent
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    agent = ExpectimaxSnakeAgent(depth=2)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=100)
    module.print_results()
"""

import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.engine import Action
from agents.base import BaseAgent

try:
    from framework.evaluation import RewardFunction
except ImportError:
    RewardFunction = None


# ═══════════════════════════════════════════════════════════════════
#  SNAKE HEURISTIC  (mschrandt/2048 ai.py)
# ═══════════════════════════════════════════════════════════════════

PERFECT_SNAKE = np.array([
    [2,    2**2,  2**3,  2**4 ],
    [2**8, 2**7,  2**6,  2**5 ],
    [2**9, 2**10, 2**11, 2**12],
    [2**16,2**15, 2**14, 2**13],
], dtype=np.float64)

INF = 2**64

_ALL_ACTIONS = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]


# ═══════════════════════════════════════════════════════════════════
#  FAST BOARD OPERATIONS  (numpy-only, no Game2048 inside search)
# ═══════════════════════════════════════════════════════════════════

def _slide_row_left(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """Slide and merge a 4-element row leftward. Returns (new_row, score)."""
    tiles = row[row != 0]
    result = np.zeros(4, dtype=np.int64)
    score = 0
    i = 0
    j = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            result[j] = merged
            score += int(merged)
            i += 2
        else:
            result[j] = tiles[i]
            i += 1
        j += 1
    return result, score


def _apply_move(board: np.ndarray, action: Action) -> Tuple[np.ndarray, bool, int]:
    """Apply a move to a 4×4 board. Returns (new_board, changed, score)."""
    new_board = np.empty((4, 4), dtype=np.int64)
    score = 0

    if action == Action.LEFT:
        for r in range(4):
            new_board[r], s = _slide_row_left(board[r])
            score += s
    elif action == Action.RIGHT:
        for r in range(4):
            rev, s = _slide_row_left(board[r, ::-1])
            new_board[r] = rev[::-1]
            score += s
    elif action == Action.UP:
        for c in range(4):
            col, s = _slide_row_left(board[:, c])
            new_board[:, c] = col
            score += s
    else:  # DOWN
        for c in range(4):
            col, s = _slide_row_left(board[::-1, c])
            new_board[::-1, c] = col
            score += s

    changed = not np.array_equal(board, new_board)
    return new_board, changed, score


# ═══════════════════════════════════════════════════════════════════
#  HEURISTIC
# ═══════════════════════════════════════════════════════════════════

def _snake_heuristic(board: np.ndarray) -> float:
    """Dot product of board values against the snake-pattern weight matrix."""
    return float(np.sum(board * PERFECT_SNAKE))


# ═══════════════════════════════════════════════════════════════════
#  EXPECTIMINIMAX SEARCH  (mschrandt/2048 ai.py)
# ═══════════════════════════════════════════════════════════════════

def _expectiminimax(board: np.ndarray, depth: float) -> float:
    """Recursive expectiminimax with integer/float depth trick.

    - depth < 0          → leaf: return snake heuristic
    - non-integer depth  → MAX node: player picks best move
    - integer depth      → CHANCE node: average over empty cells (2-tile spawn)

    depth decrements by 0.5, alternating chance/player each level.
    """
    if depth < 0:
        return _snake_heuristic(board)

    if depth != int(depth):
        # ── MAX node ──
        best = -INF
        for action in _ALL_ACTIONS:
            new_board, changed, _ = _apply_move(board, action)
            if changed:
                val = _expectiminimax(new_board, depth - 0.5)
                if val > best:
                    best = val
        return best

    else:
        # ── CHANCE node ──
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) == 0:
            return _snake_heuristic(board)

        n_empty = len(empty_cells)
        total = 0.0
        for (r, c) in empty_cells:
            child = board.copy()
            child[r, c] = 2
            total += _expectiminimax(child, depth - 0.5)
        return total / n_empty


# ═══════════════════════════════════════════════════════════════════
#  AGENT
# ═══════════════════════════════════════════════════════════════════

class ExpectimaxSnakeAgent(BaseAgent):
    """Expectimax agent using the mschrandt/2048 snake heuristic.

    Args:
        depth: Search depth. depth=2 = 3 chance + 2 player levels.
               Default 2 (matches mschrandt/2048 repo).

    Example::

        agent = ExpectimaxSnakeAgent(depth=2)
    """

    agent_type = "expectimax_snake"

    def __init__(self, depth: int = 2):
        super().__init__(f"ExpectimaxSnake(d={depth})")
        self.depth = depth

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> Action:
        if len(available_moves) == 1:
            return available_moves[0]

        board = state.astype(np.int64)
        best_action = available_moves[0]
        best_value  = -INF

        # Apply move first, then call expectiminimax at a CHANCE node.
        for move in available_moves:
            new_board, changed, _ = _apply_move(board, move)
            if not changed:
                continue
            val = _expectiminimax(new_board, self.depth)
            if val >= best_value:
                best_value  = val
                best_action = move

        return best_action

    def get_params(self) -> Dict:
        return {'depth': self.depth}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    agent  = ExpectimaxSnakeAgent(depth=2)
    logger = RunLogger(log_move_detail=True)

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
        print_board=False,
        num_workers=13,
    )
    module.run(num_games=100)
    module.print_results()
