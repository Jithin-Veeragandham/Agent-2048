"""
expectimax.py
=============

Expectimax agent for 2048, ported from cczhong11/2048-ai.

Logic mirrors ExpectedMax.py from that repo, adapted to use our
numpy board representation instead of Python lists + deepcopy.

Key design (from the reference):
    - Lookup-table heuristic: heurs_table[65536] pre-computed at startup.
      Each 4-tile row is encoded as a 16-bit nibble key → O(1) eval.
      Weights: total_sum×5 + empty×2 + merge×1 − monotonicity.
      Applied to all 4 rows and all 4 columns (transposed), weighted
      by row position (×4×(row_index+1)).
    - Search: get_expected (CHANCE) ↔ try_move (MAX), depth-limited.
      Spawns 2-tiles (×0.9) and 4-tiles (×0.1) across all empty cells.
    - depth=2 matches the reference default.

Usage::

    from agents.expectimax import ExpectimaxAgent
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    agent = ExpectimaxAgent(depth=2)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=10)
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
#  LOOKUP TABLE HEURISTIC  (cczhong11/2048-ai ExpectedMax.py)
# ═══════════════════════════════════════════════════════════════════

# Maps tile value → log2 nibble (e.g. 0→0, 2→1, 4→2, 8→3, …, 65536→16)
_TILE_TO_NIBBLE: Dict[int, int] = {1 << i: i for i in range(17)}
_TILE_TO_NIBBLE[0] = 0

# Pre-computed heuristic score for every possible 16-bit row encoding
_heurs_table: List[float] = [0.0] * 65536

INF = 2**64


def _build_heurs_table() -> None:
    """Build heurs_table once at import time (matches build_table() in reference)."""
    sw, ew, mw, mono_w = 5, 2, 1, 1
    for row in range(65536):
        num = [row >> 12 & 15, row >> 8 & 15, row >> 4 & 15, row & 15]
        total_sum = sum(2 ** n for n in num)
        empty = sum(1 for n in num if n == 0)
        merge = sum(2 ** num[i] for i in range(1, 4)
                    if num[i] == num[i - 1] and num[i] != 0)
        mono_left = mono_right = 0
        for i in range(1, 4):
            tmp = num[i - 1] - num[i]
            if tmp > 0:
                mono_left  += 2 ** tmp
            else:
                mono_right += 2 ** (-tmp)
        _heurs_table[row] = (total_sum * sw + ew * empty + mw * merge
                             - mono_w * min(mono_left, mono_right))


_build_heurs_table()


# ═══════════════════════════════════════════════════════════════════
#  FAST BOARD OPERATIONS  (numpy-only, no Game2048 inside search)
# ═══════════════════════════════════════════════════════════════════

_ALL_ACTIONS = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]


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
#  HEURISTIC  (cczhong11/2048-ai get_score)
# ═══════════════════════════════════════════════════════════════════

def _get_score(board: np.ndarray) -> float:
    """Evaluate board via lookup table on rows + transposed columns.

    Each row is encoded as a 16-bit nibble key (log2 tile values),
    looked up in _heurs_table, and weighted by row position.
    Mirrors get_score() from ExpectedMax.py.
    """
    total = 0.0
    for j in range(4):
        row = board[j]
        key = ((_TILE_TO_NIBBLE[int(row[3])] << 12) |
               (_TILE_TO_NIBBLE[int(row[2])] << 8)  |
               (_TILE_TO_NIBBLE[int(row[1])] << 4)  |
                _TILE_TO_NIBBLE[int(row[0])])
        total += _heurs_table[key] * 4 * (j + 1)
    for j in range(4):
        col = board[:, j]
        key = ((_TILE_TO_NIBBLE[int(col[3])] << 12) |
               (_TILE_TO_NIBBLE[int(col[2])] << 8)  |
               (_TILE_TO_NIBBLE[int(col[1])] << 4)  |
                _TILE_TO_NIBBLE[int(col[0])])
        total += _heurs_table[key] * 4 * (j + 1)
    return total


# ═══════════════════════════════════════════════════════════════════
#  EXPECTIMAX SEARCH  (cczhong11/2048-ai get_expected / try_move)
# ═══════════════════════════════════════════════════════════════════

def _try_move(board: np.ndarray, depth: int) -> float:
    """MAX node: try all 4 moves, return best expected score.

    Mirrors try_move() from ExpectedMax.py.
    """
    best = 0.0
    any_valid = False
    for action in _ALL_ACTIONS:
        new_board, changed, _ = _apply_move(board, action)
        if changed:
            val = _get_expected(new_board, depth)
            if val > best:
                best = val
            any_valid = True
    return best if any_valid else 0.0


def _get_expected(board: np.ndarray, depth: int) -> float:
    """CHANCE node: average over all empty cells spawning 2 (×0.9) or 4 (×0.1).

    Mirrors get_expected() from ExpectedMax.py exactly, including the
    cumulative-tile behaviour of the reference (board2/board4 copied
    once and mutated across iterations).
    """
    empty_cells = np.argwhere(board == 0)
    n_empty = len(empty_cells)

    if depth == 0:
        return _get_score(board)

    if n_empty == 0:
        return _try_move(board, depth - 1)

    board2 = board.copy()   # accumulates 2-tiles across iterations (matches reference)
    board4 = board.copy()   # accumulates 4-tiles across iterations (matches reference)
    total = 0.0
    for (r, c) in empty_cells:
        board2[r, c] = 2
        total += _try_move(board2, depth - 1) * 0.9
        board4[r, c] = 4
        total += _try_move(board4, depth - 1) * 0.1

    return total / n_empty


# ═══════════════════════════════════════════════════════════════════
#  AGENT
# ═══════════════════════════════════════════════════════════════════

class ExpectimaxAgent(BaseAgent):
    """Expectimax agent for 2048, ported from cczhong11/2048-ai.

    Uses a lookup-table heuristic (total_sum×5 + empty×2 + merge×1
    − monotonicity) and get_expected/try_move search with 90/10
    2-tile/4-tile spawn probabilities.

    Args:
        depth: Search depth (chance node levels). depth=2 matches
               the reference default.

    Example::

        agent = ExpectimaxAgent(depth=2)
    """

    agent_type = "expectimax"

    def __init__(self, depth: int = 2):
        super().__init__(f"Expectimax(d={depth})")
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

        # Mirror get_move: apply each move then call get_expected (CHANCE node).
        for move in available_moves:
            new_board, changed, _ = _apply_move(board, move)
            if not changed:
                continue
            val = _get_expected(new_board, self.depth)
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

    agent  = ExpectimaxAgent(depth=2)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
        print_board=False,
        num_workers=4,
    )
    module.run(num_games=10)
    module.print_results()
