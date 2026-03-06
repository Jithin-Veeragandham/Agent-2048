"""
mcts.py
=======

Monte Carlo Tree Search agents for 2048.

Two variants:
    - MCTSAgent:          Classic MCTS with random rollouts.
    - MCTSHeuristicAgent: Optimized MCTS — heuristic-guided rollouts
                          with game score as tree value (matching the
                          approach from thomasahle/mcts-2048 and
                          chadpalmer2/2048_mcts).

Key design: the heuristic guides rollout MOVE SELECTION, but the
value backpropagated through the tree is the raw game score — not
the heuristic value. This gives UCB1 large, well-separated values
to work with, while the heuristic steers simulations toward good
board states.

Usage::

    from mcts import MCTSAgent, MCTSHeuristicAgent
    from interaction import InteractionModule
    from utils import RunLogger

    agent = MCTSHeuristicAgent(num_simulations=500, rollout_depth=10)

    logger = RunLogger()
    module = InteractionModule(
        config={"grid_size": 4, "random_seed": 42},
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=1)
    module.print_results()
"""

import math
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from game import Game2048, Action

try:
    from evaluation import RewardFunction
except ImportError:
    RewardFunction = None

try:
    from interaction import BaseAgent
except ImportError:
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
#  HEURISTIC EVALUATION (for rollout move selection only)
# ═══════════════════════════════════════════════════════════════════

def _default_heuristic(board: np.ndarray) -> float:
    """Composite heuristic for guiding rollout move selection.

    Fully vectorized — no Python loops. Used to pick which move
    to play during a rollout, NOT as the tree value.

    Components: tile_score, empty_bonus, monotonicity (snake pattern),
    merge_potential, smoothness penalty.
    """
    rows, cols = board.shape
    n = rows * cols

    # ── Tile score ────────────────────────────────────────────
    non_zero = board[board > 0].astype(float)
    tile = float(np.sum(non_zero * np.log2(non_zero))) if len(non_zero) > 0 else 0.0

    # ── Empty bonus (vectorized adjacency) ────────────────────
    empty_mask = (board == 0)
    empty_count = int(np.sum(empty_mask))
    horiz = int(np.sum(empty_mask[:, :-1] & empty_mask[:, 1:]))
    vert = int(np.sum(empty_mask[:-1, :] & empty_mask[1:, :]))
    empty = float(empty_count ** 2 + horiz + vert)

    # ── Monotonicity (snake pattern) ──────────────────────────
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

    # ── Merge potential (vectorized) ──────────────────────────
    h_match = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] > 0)
    v_match = (board[:-1, :] == board[1:, :]) & (board[:-1, :] > 0)
    merge = float(np.sum(log_board[:, :-1][h_match]) + np.sum(log_board[:-1, :][v_match]))

    # ── Smoothness (vectorized) ───────────────────────────────
    h_both = (board[:, :-1] > 0) & (board[:, 1:] > 0)
    v_both = (board[:-1, :] > 0) & (board[1:, :] > 0)
    smooth = float(
        np.sum(np.abs(log_board[:, :-1] - log_board[:, 1:])[h_both])
        + np.sum(np.abs(log_board[:-1, :] - log_board[1:, :])[v_both])
    )

    # ── Normalize & combine ───────────────────────────────────
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


def _tree_value(game: Game2048) -> float:
    """Compute the value to backpropagate through the MCTS tree.

    Returns the raw game score — matching poomstas/2048_MCTS's
    state.get_result() which returns the cumulative merge score.

    Simple, fast, and gives UCB1 well-separated values in the
    thousands range.
    """
    return float(game.get_score())


# ═══════════════════════════════════════════════════════════════════
#  SHARED: resolve rollout policy function
# ═══════════════════════════════════════════════════════════════════

def _resolve_rollout_policy(eval_fn=None, reward_fn=None):
    """Pick the best available rollout policy function.

    This function is used to SELECT MOVES during rollouts, not
    to compute tree values.
    """
    if reward_fn is not None:
        return reward_fn.compute
    if eval_fn is not None:
        return eval_fn
    if RewardFunction is not None:
        rf = RewardFunction(weights={
            'tile': 1.0, 'empty': 0.5, 'merge': 0.5, 'smooth': 0.3,
        })
        return rf.compute
    return _default_heuristic


# ═══════════════════════════════════════════════════════════════════
#  PARALLEL ROLLOUT WORKERS (top-level for pickling)
# ═══════════════════════════════════════════════════════════════════

def _worker_random_rollout(board, score, rollout_depth, tile_2_prob):
    """Random rollout worker. Returns game score."""
    sim = Game2048.from_state(board, score=score, config={
        'tile_2_probability': tile_2_prob,
    })
    for _ in range(rollout_depth):
        if sim.is_game_over():
            break
        moves = sim.get_available_moves()
        if not moves:
            break
        sim.move(moves[np.random.randint(len(moves))])
    return float(sim.get_score())


def _worker_greedy_rollout(board, score, rollout_depth, tile_2_prob):
    """Greedy heuristic rollout worker. Returns game score + tile sum."""
    sim = Game2048.from_state(board, score=score, config={
        'tile_2_probability': tile_2_prob,
    })
    for _ in range(rollout_depth):
        if sim.is_game_over():
            break
        moves = sim.get_available_moves()
        if not moves:
            break

        best_h = -float('inf')
        best_move = moves[0]
        for move in moves:
            child = sim.clone()
            child.move(move)
            h = _default_heuristic(child.get_state())
            if h > best_h:
                best_h = h
                best_move = move

        sim.move(best_move)

    return sim.get_score() + float(np.sum(sim.get_state()))


# ═══════════════════════════════════════════════════════════════════
#  MCTS NODE (shared by both agents)
# ═══════════════════════════════════════════════════════════════════

class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = ('game', 'parent', 'action', 'children',
                 'visits', 'value', 'untried')

    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children: Dict[Action, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.untried: List[Action] = game.get_available_moves()

    @property
    def is_fully_expanded(self):
        return len(self.untried) == 0

    @property
    def is_terminal(self):
        return self.game.is_game_over() or not self.game.get_available_moves()

    def ucb1(self, c):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c):
        return max(self.children.values(), key=lambda ch: ch.ucb1(c))

    def expand(self):
        action = self.untried.pop()
        child_game = self.game.clone()
        child_game.move(action)
        child = MCTSNode(child_game, parent=self, action=action)
        self.children[action] = child
        return child


# ═══════════════════════════════════════════════════════════════════
#  CLASSIC MCTS — random rollouts
# ═══════════════════════════════════════════════════════════════════

class MCTSAgent(BaseAgent):
    """Classic MCTS with random rollouts.

    Rollouts play random moves, tree values use raw game score.

    Args:
        num_simulations: MCTS iterations per move decision.
        rollout_depth:   Max random moves per rollout.
        exploration:     UCB1 exploration constant.
        num_workers:     Parallel rollout workers. 1 = sequential.

    Example::

        agent = MCTSAgent(num_simulations=500, rollout_depth=20)
    """

    def __init__(
        self,
        num_simulations: int = 200,
        rollout_depth: int = 10,
        exploration: float = math.sqrt(2),
        num_workers: int = 1,
    ):
        super().__init__(f"MCTS(n={num_simulations},d={rollout_depth})")
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.num_workers = num_workers

    def _select(self, node):
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.exploration)
        return node

    def _rollout(self, game):
        """Random rollout, return game score + tile sum."""
        sim = game.clone()
        for _ in range(self.rollout_depth):
            if sim.is_game_over():
                break
            moves = sim.get_available_moves()
            if not moves:
                break
            sim.move(moves[np.random.randint(len(moves))])
        return _tree_value(sim)

    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def choose_action(self, state, available_moves, game_context=None):
        if len(available_moves) == 1:
            return available_moves[0]

        game = game_context['game'] if game_context and 'game' in game_context else None
        if game is None:
            game = Game2048.from_state(state)

        root = MCTSNode(game.clone())

        if self.num_workers <= 1:
            for _ in range(self.num_simulations):
                node = self._select(root)
                if not node.is_terminal and not node.is_fully_expanded:
                    node = node.expand()
                value = self._rollout(node.game)
                self._backpropagate(node, value)
        else:
            self._run_parallel(root, game)

        if not root.children:
            return available_moves[0]

        return max(root.children.values(), key=lambda c: c.visits).action

    def _run_parallel(self, root, game):
        tile_2_prob = game.tile_2_prob
        sims_done = 0
        batch_size = self.num_workers * 2

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            while sims_done < self.num_simulations:
                batch_nodes = []
                current_batch = min(batch_size, self.num_simulations - sims_done)

                for _ in range(current_batch):
                    node = self._select(root)
                    if not node.is_terminal and not node.is_fully_expanded:
                        node = node.expand()
                    batch_nodes.append(node)

                futures = {}
                for i, node in enumerate(batch_nodes):
                    f = pool.submit(
                        _worker_random_rollout,
                        node.game.get_state(), node.game.get_score(),
                        self.rollout_depth, tile_2_prob,
                    )
                    futures[f] = i

                results = [None] * len(batch_nodes)
                for f in as_completed(futures):
                    results[futures[f]] = f.result()

                for node, value in zip(batch_nodes, results):
                    self._backpropagate(node, value)

                sims_done += current_batch


# ═══════════════════════════════════════════════════════════════════
#  HEURISTIC MCTS — heuristic guides rollout, score drives tree
# ═══════════════════════════════════════════════════════════════════

class MCTSHeuristicAgent(BaseAgent):
    """Optimized MCTS: heuristic-guided rollouts, game-score tree values.

    The heuristic picks which move to play at each rollout step
    (greedy move selection). But the value backpropagated through
    the MCTS tree is the raw game score + tile sum — giving UCB1
    large, well-separated values to distinguish good from bad branches.

    This matches the approach from thomasahle/mcts-2048 (greedy
    rollout policy + SumMeasure for evaluation) and chadpalmer2
    (game score for evaluation).

    When rollout_depth=0, the leaf node's game score + tile sum is
    used directly (no rollout).

    Args:
        num_simulations: MCTS iterations per move decision.
        rollout_depth:   Greedy rollout steps. 0 = direct score eval.
        exploration:     UCB1 exploration constant.
        num_workers:     Parallel rollout workers. 1 = sequential.
        eval_fn:         Optional heuristic for rollout move selection.
        reward_fn:       Optional RewardFunction for rollout move selection.

    Example::

        # Greedy rollout, sequential
        agent = MCTSHeuristicAgent(num_simulations=500, rollout_depth=10)

        # Direct score eval, parallel
        agent = MCTSHeuristicAgent(
            num_simulations=1000, rollout_depth=0, num_workers=4,
        )
    """

    def __init__(
        self,
        num_simulations: int = 200,
        rollout_depth: int = 10,
        exploration: float = math.sqrt(2),
        num_workers: int = 1,
        eval_fn=None,
        reward_fn=None,
    ):
        depth_label = f"d={rollout_depth}" if rollout_depth > 0 else "direct"
        workers_label = f",w={num_workers}" if num_workers > 1 else ""
        super().__init__(
            f"MCTS-H(n={num_simulations},{depth_label}{workers_label})"
        )
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.num_workers = num_workers
        # Heuristic is ONLY for rollout move selection
        self.rollout_policy = _resolve_rollout_policy(eval_fn, reward_fn)

    def _select(self, node):
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.exploration)
        return node

    def _rollout(self, game):
        """Greedy heuristic rollout, return game score + tile sum."""
        if self.rollout_depth == 0:
            return _tree_value(game)

        sim = game.clone()
        for _ in range(self.rollout_depth):
            if sim.is_game_over():
                break
            moves = sim.get_available_moves()
            if not moves:
                break

            # Heuristic picks the move
            best_h = -float('inf')
            best_move = moves[0]
            for move in moves:
                child = sim.clone()
                child.move(move)
                h = self.rollout_policy(child.get_state())
                if h > best_h:
                    best_h = h
                    best_move = move

            sim.move(best_move)

        # Game score + tile sum drives the tree
        return _tree_value(sim)

    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def choose_action(self, state, available_moves, game_context=None):
        if len(available_moves) == 1:
            return available_moves[0]

        game = game_context['game'] if game_context and 'game' in game_context else None
        if game is None:
            game = Game2048.from_state(state)

        root = MCTSNode(game.clone())

        if self.num_workers <= 1:
            for _ in range(self.num_simulations):
                node = self._select(root)
                if not node.is_terminal and not node.is_fully_expanded:
                    node = node.expand()
                value = self._rollout(node.game)
                self._backpropagate(node, value)
        else:
            self._run_parallel(root, game)

        if not root.children:
            return available_moves[0]

        return max(root.children.values(), key=lambda c: c.visits).action

    def _run_parallel(self, root, game):
        tile_2_prob = game.tile_2_prob
        sims_done = 0
        batch_size = self.num_workers * 2

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            while sims_done < self.num_simulations:
                batch_nodes = []
                current_batch = min(batch_size, self.num_simulations - sims_done)

                for _ in range(current_batch):
                    node = self._select(root)
                    if not node.is_terminal and not node.is_fully_expanded:
                        node = node.expand()
                    batch_nodes.append(node)

                futures = {}
                for i, node in enumerate(batch_nodes):
                    f = pool.submit(
                        _worker_greedy_rollout,
                        node.game.get_state(), node.game.get_score(),
                        self.rollout_depth, tile_2_prob,
                    )
                    futures[f] = i

                results = [None] * len(batch_nodes)
                for f in as_completed(futures):
                    results[futures[f]] = f.result()

                for node, value in zip(batch_nodes, results):
                    self._backpropagate(node, value)

                sims_done += current_batch


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from interaction import InteractionModule
    from utils import RunLogger

    config = {"grid_size": 4, "random_seed": 42}
    logger = RunLogger()

    # ── Run classic MCTS ──────────────────────────────────────
    # print("=" * 60)
    # print("  CLASSIC MCTS (random rollouts)")
    # print("=" * 60)
    # agent_classic = MCTSAgent(num_simulations=500, rollout_depth=20)
    # module = InteractionModule(
    #     config=config, agent=agent_classic,
    #     logger=logger, verbose=True, print_board=True,
    # )
    # module.run(num_games=1)
    # module.print_results()

    # ── Run heuristic MCTS ────────────────────────────────────
    print("=" * 60)
    print("  HEURISTIC MCTS (greedy rollouts, game score tree value)")
    print("=" * 60)
    agent_heuristic = MCTSAgent(num_simulations=500, rollout_depth=50, exploration=100.0)
    module2 = InteractionModule(
        config=config, agent=agent_heuristic,
        logger=logger, verbose=True, print_board=True,
    )
    module2.run(num_games=10)
    module2.print_results()