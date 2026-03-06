"""
beam_search.py
==============

Heuristic Beam Search agent for 2048.

Implements a breadth-limited tree search that keeps the top-k
(beam_width) most promising board states at each depth level,
scored by a composite heuristic evaluation function.

Usage (Jupyter notebook)::

    from beam_search import BeamSearchAgent
    from interaction import InteractionModule
    from utils import RunLogger

    agent = BeamSearchAgent(beam_width=10, search_depth=10)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4, "random_seed": 42},
        agent=agent,
        logger=logger,
        verbose=True,
        print_board=True,
    )
    module.run(num_games=1)
    module.print_results()
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from game import Game2048, Action

# Try importing from the project's evaluation module; fall back to
# a self-contained heuristic if evaluation.py isn't available.
try:
    from evaluation import RewardFunction
except ImportError:
    RewardFunction = None

try:
    from interaction import BaseAgent
except ImportError:
    # Minimal stub so the file works standalone
    from abc import ABC, abstractmethod

    class BaseAgent(ABC):
        def __init__(self, name: str):
            self.name = name

        @abstractmethod
        def choose_action(self, state, available_moves, game_context=None):
            ...

        def on_episode_start(self): pass
        def on_episode_end(self, final_state, score): pass
        def on_move_result(self, state, action, reward, next_state, done): pass


# ═══════════════════════════════════════════════════════════════════
#  HEURISTIC EVALUATION (self-contained fallback)
# ═══════════════════════════════════════════════════════════════════

def _default_heuristic(board: np.ndarray) -> float:
    """Composite heuristic for evaluating a 2048 board state.

    Components:
        - tile_score:      value * log2(value) for each tile
        - empty_bonus:     (empty_count)^2 + adjacent empty pairs
        - monotonicity:    corner-gradient snake pattern reward
        - merge_potential: weighted count of adjacent equal pairs
        - smoothness:      penalty for large neighbor value gaps

    Returns:
        float: Composite score (higher = better board).
    """
    rows, cols = board.shape
    n = rows * cols

    # ── Tile score ────────────────────────────────────────────
    non_zero = board[board > 0].astype(float)
    tile = float(np.sum(non_zero * np.log2(non_zero))) if len(non_zero) > 0 else 0.0

    # ── Empty bonus ───────────────────────────────────────────
    empty_mask = (board == 0)
    empty_count = int(np.sum(empty_mask))
    adjacency = 0
    for r in range(rows):
        for c in range(cols):
            if empty_mask[r, c]:
                if c + 1 < cols and empty_mask[r, c + 1]:
                    adjacency += 1
                if r + 1 < rows and empty_mask[r + 1, c]:
                    adjacency += 1
    empty = float(empty_count ** 2 + adjacency)

    # ── Monotonicity (snake pattern from top-left corner) ─────
    # Build ideal weight grid: highest weight at corner, decaying
    # along a snake path.
    weights = np.zeros((rows, cols), dtype=float)
    idx = 0
    for r in range(rows):
        row_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in row_range:
            weights[r, c] = 4.0 ** (n - 1 - idx)
            idx += 1

    log_board = np.zeros_like(board, dtype=float)
    mask = board > 0
    log_board[mask] = np.log2(board[mask].astype(float))
    mono = float(np.sum(log_board * weights / np.max(weights)))

    # ── Merge potential ───────────────────────────────────────
    merge = 0.0
    for r in range(rows):
        for c in range(cols):
            val = board[r, c]
            if val == 0:
                continue
            w = np.log2(val)
            if c + 1 < cols and board[r, c + 1] == val:
                merge += w
            if r + 1 < rows and board[r + 1, c] == val:
                merge += w

    # ── Smoothness (penalty) ──────────────────────────────────
    smooth = 0.0
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0:
                continue
            if c + 1 < cols and board[r, c + 1] != 0:
                smooth += abs(log_board[r, c] - log_board[r, c + 1])
            if r + 1 < rows and board[r + 1, c] != 0:
                smooth += abs(log_board[r, c] - log_board[r + 1, c])

    # ── Normalization & combination ───────────────────────────
    t_norm = tile / max(2048 * 11 * n, 1)
    e_norm = empty / max(n ** 2 + n * 2, 1)
    m_norm = mono / max(11 * n, 1)
    mp_norm = merge / max(11 * n, 1)
    s_norm = smooth / max(11 * n * 2, 1)

    return (
        1.0 * t_norm
        + 0.5 * e_norm
        + 1.0 * m_norm
        + 0.5 * mp_norm
        - 0.3 * s_norm
    )


# ═══════════════════════════════════════════════════════════════════
#  BEAM SEARCH AGENT
# ═══════════════════════════════════════════════════════════════════

class BeamSearchAgent(BaseAgent):
    """Heuristic Beam Search agent for 2048.

    At each game step the agent:
        1. Expands every candidate state by all valid moves.
        2. Scores the resulting states with a heuristic function.
        3. Keeps only the top-k (beam_width) states.
        4. Repeats for search_depth levels.
        5. Picks the root action that led to the best leaf.

    Random tile spawns during simulation are sampled once per
    expansion (single-sample approximation) to keep the search
    tractable. For more robust evaluation, increase beam_width
    rather than search_depth.

    Args:
        beam_width:   Number of candidate states to keep at each level.
        search_depth: How many moves to look ahead.
        eval_fn:      Optional callable(np.ndarray) -> float. If None,
                      uses the built-in composite heuristic.
        reward_fn:    Optional RewardFunction from evaluation.py.
                      Overrides eval_fn if provided.

    Example::

        agent = BeamSearchAgent(beam_width=15, search_depth=20)
    """

    def __init__(
        self,
        beam_width: int = 15,
        search_depth: int = 20,
        eval_fn=None,
        reward_fn=None,
    ):
        super().__init__(f"BeamSearch(w={beam_width},d={search_depth})")
        self.beam_width = beam_width
        self.search_depth = search_depth

        # Resolve evaluation function
        if reward_fn is not None:
            self.eval_fn = reward_fn.compute
        elif eval_fn is not None:
            self.eval_fn = eval_fn
        elif RewardFunction is not None:
            # Use project's RewardFunction with monotonicity-friendly weights
            self._reward_fn = RewardFunction(weights={
                'tile': 1.0, 'empty': 0.5, 'merge': 0.5, 'smooth': 0.3,
            })
            self.eval_fn = self._reward_fn.compute
        else:
            self.eval_fn = _default_heuristic

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Select the best action via beam search.

        Args:
            state:           Current board (unused directly; we clone the game).
            available_moves: Valid actions from current state.
            game_context:    Must contain 'game' key with a Game2048 instance.

        Returns:
            Action: The move whose search tree produced the highest-scoring leaf.
        """
        if len(available_moves) == 1:
            return available_moves[0]

        game = game_context['game'] if game_context and 'game' in game_context else None

        if game is None:
            # Fallback: reconstruct from state (no score info)
            game = Game2048.from_state(state)

        # ── Initialize beam: expand root into children ────────
        # Each beam entry: (score_of_leaf, root_action, Game2048_clone)
        beam: List[Tuple[float, Action, Game2048]] = []

        for action in available_moves:
            sim = game.clone()
            valid, _ = sim.move(action)
            if valid:
                score = self.eval_fn(sim.get_state())
                beam.append((score, action, sim))

        if not beam:
            return available_moves[0]

        # ── Expand beam for search_depth levels ───────────────
        for depth in range(1, self.search_depth):
            candidates: List[Tuple[float, Action, Game2048]] = []

            for _, root_action, sim_game in beam:
                moves = sim_game.get_available_moves()
                if not moves:
                    # Terminal state — carry forward with current score
                    score = self.eval_fn(sim_game.get_state())
                    candidates.append((score, root_action, sim_game))
                    continue

                for move in moves:
                    child = sim_game.clone()
                    valid, _ = child.move(move)
                    if valid:
                        score = self.eval_fn(child.get_state())
                        candidates.append((score, root_action, child))

            if not candidates:
                break

            # Keep top-k by heuristic score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[: self.beam_width]

        # ── Pick the root action that led to the best leaf ────
        best = max(beam, key=lambda x: x[0])
        return best[1]


# ═══════════════════════════════════════════════════════════════════
#  MAIN — run directly: python beam_search.py
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from interaction import InteractionModule
    from utils import RunLogger

    config = {"grid_size": 4, "random_seed": 42}
    agent = BeamSearchAgent(beam_width=5, search_depth=5)
    logger = RunLogger()

    module = InteractionModule(
        config=config,
        agent=agent,
        logger=logger,
        verbose=True,
        print_board=True,
    )
    module.run(num_games=1)
    module.print_results()