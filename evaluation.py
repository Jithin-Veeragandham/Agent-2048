import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple

class RewardFunction:
    """Composite reward function for evaluating 2048 board states.

    Combines five components to capture both tile progress and
    board structure quality.

    Components:
        - tile_score:      Weights higher tiles exponentially more
        - empty_bonus:     Rewards open + clustered empty cells
        - monotonicity:    Rewards decreasing gradient from corner (snake pattern)
        - merge_potential: Rewards adjacent same-value tile pairs
        - smoothness:      Penalizes large value gaps between neighbors

    Args:
        weights: Dict with keys 'tile', 'empty', 'mono', 'merge', 'smooth'.
            Defaults are tuned for a 4×4 board. Override for experimentation.

    Example::

        rf = RewardFunction()
        score = rf.compute(game.get_state())

        # Custom weights for search agents
        rf_search = RewardFunction(weights={
            'tile': 1.0, 'empty': 0.5, 'mono': 1.0,
            'merge': 0.5, 'smooth': 0.3
        })
    """

    DEFAULT_WEIGHTS = {
        'tile':   1.0,
        'empty':  0.5,
        'merge':  0.5,
        'smooth': 0.3,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}

    # ─── Individual Components ────────────────────────────────────

    @staticmethod
    def tile_score(board: np.ndarray) -> float:
        """Weighted tile score: value × log₂(value) for each tile.

        Higher tiles get disproportionate credit since they are
        exponentially harder to create (a 2048 needs 11 successive merges).

        Args:
            board: 2D numpy array of tile values.

        Returns:
            float: Sum of value * log2(value) for all non-zero tiles.
        """
        non_zero = board[board > 0].astype(float)
        if len(non_zero) == 0:
            return 0.0
        return float(np.sum(non_zero * np.log2(non_zero)))

    @staticmethod
    def empty_bonus(board: np.ndarray) -> float:
        """Rewards open and clustered empty cells.

        Two sub-components:
        - Count: (empty_cells)² — captures urgency when board is nearly full
        - Adjacency: bonus for each pair of neighboring empty cells

        Args:
            board: 2D numpy array of tile values.

        Returns:
            float: Combined empty cell score.
        """
        empty_mask = (board == 0)
        empty_count = int(np.sum(empty_mask))

        # Count component: squared to emphasize scarcity
        count_score = empty_count ** 2

        # Adjacency component: count neighboring empty-empty pairs
        adjacency = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if empty_mask[r, c]:
                    if c + 1 < cols and empty_mask[r, c + 1]:
                        adjacency += 1
                    if r + 1 < rows and empty_mask[r + 1, c]:
                        adjacency += 1

        return float(count_score + adjacency)

    

    @staticmethod
    def merge_potential(board: np.ndarray) -> float:
        """Rewards adjacent same-value tile pairs, weighted by tile value.

        Adjacent 512-512 pairs are more valuable than adjacent 2-2 pairs
        since they represent a much bigger scoring opportunity.

        Args:
            board: 2D numpy array of tile values.

        Returns:
            float: Weighted count of mergeable adjacent pairs.
        """
        score = 0.0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                val = board[r, c]
                if val == 0:
                    continue
                weight = np.log2(val) if val > 0 else 0
                # Check right neighbor
                if c + 1 < cols and board[r, c + 1] == val:
                    score += weight
                # Check bottom neighbor
                if r + 1 < rows and board[r + 1, c] == val:
                    score += weight
        return score

    @staticmethod
    def smoothness(board: np.ndarray) -> float:
        """Penalizes large value gaps between neighboring tiles.

        Computes sum of |log₂(tᵢ) - log₂(tⱼ)| for all adjacent
        non-zero tile pairs. A smooth board is easier to merge.

        Args:
            board: 2D numpy array of tile values.

        Returns:
            float: Smoothness penalty (lower = smoother board).
        """
        log_board = np.zeros_like(board, dtype=float)
        mask = board > 0
        log_board[mask] = np.log2(board[mask].astype(float))

        penalty = 0.0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == 0:
                    continue
                # Right neighbor
                if c + 1 < cols and board[r, c + 1] != 0:
                    penalty += abs(log_board[r, c] - log_board[r, c + 1])
                # Bottom neighbor
                if r + 1 < rows and board[r + 1, c] != 0:
                    penalty += abs(log_board[r, c] - log_board[r + 1, c])
        return penalty

    # ─── Composite Score ──────────────────────────────────────────

    def compute(self, board: np.ndarray) -> float:
        """Compute the full composite reward for a board state.

        R(s) = α·tile + β·empty + γ·mono + δ·merge - λ·smooth

        Each component is normalized to [0, 1] range before weighting
        to prevent any single term from dominating.

        Args:
            board: 2D numpy array of tile values.

        Returns:
            float: Composite reward score.
        """
        w = self.weights

        t = self.tile_score(board)
        e = self.empty_bonus(board)
        mp = self.merge_potential(board)
        s = self.smoothness(board)

        # Normalization constants for a 4×4 board
        # These are approximate upper bounds for each component
        max_tile = board.shape[0] ** 2
        t_norm = max(t / (2048 * 11 * max_tile), 1e-8)    # tile_score upper bound
        e_norm = max(e / (max_tile ** 2 + max_tile * 2), 1e-8)  # empty_bonus upper bound
        mp_norm = max(mp / (11 * max_tile), 1e-8)          # merge upper bound
        s_norm = max(s / (11 * max_tile * 2), 1e-8)        # smooth upper bound

        return (
            w['tile']   * t_norm
            + w['empty']  * e_norm
            + w['merge']  * mp_norm
            - w['smooth'] * s_norm
        )

    def compute_breakdown(self, board: np.ndarray) -> Dict[str, float]:
        """Return individual component scores for debugging/analysis.

        Args:
            board: 2D numpy array of tile values.

        Returns:
            Dict with raw values for each component plus the composite.
        """
        return {
            'tile_score':      self.tile_score(board),
            'empty_bonus':     self.empty_bonus(board),
            'merge_potential': self.merge_potential(board),
            'smoothness':      self.smoothness(board),
            'composite':       self.compute(board),
        }


# ═══════════════════════════════════════════════════════════════════
#  PRESET REWARD CONFIGS — per agent type
# ═══════════════════════════════════════════════════════════════════

# Full shaping for search agents (MCTS, Beam Search)
REWARD_SEARCH = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.5, 'mono': 1.0, 'merge': 0.5, 'smooth': 0.3
})

# Simplified for RL agents (DQN, PPO) — let the network learn structure
REWARD_RL = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.3, 'mono': 0.0, 'merge': 0.0, 'smooth': 0.0
})

# Minimal shaping for AlphaZero — value head learns evaluation
'''REWARD_ALPHAZERO = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.1, 'mono': 0.0, 'merge': 0.0, 'smooth': 0.0
})'''


# ═══════════════════════════════════════════════════════════════════
#  GAME EVALUATOR — used after games to compare agents
# ═══════════════════════════════════════════════════════════════════

class EpisodeStats:
    """Stats collected for a single game episode.

    Attributes:
        score:         Final merge score
        highest_tile:  Maximum tile value on the board at game end
        move_count:    Total number of valid moves made
        reached_2048:  Whether the 2048 tile was achieved
        reached_4096:  Whether the 4096 tile was achieved
        reached_8192:  Whether the 8192 tile was achieved
        tile_dist:     Dict mapping tile values to counts on final board
        move_times_ms: List of inference times (ms) for each move
    """

    def __init__(self):
        self.score: int = 0
        self.highest_tile: int = 0
        self.move_count: int = 0
        self.reached_2048: bool = False
        self.reached_4096: bool = False
        self.reached_8192: bool = False
        self.tile_dist: Dict[int, int] = {}
        self.move_times_ms: List[float] = []


class GameEvaluator:
    """Collects and aggregates evaluation metrics across many game episodes.

    Tracks both performance metrics (win rate, score, tile distribution)
    and efficiency metrics (inference time per move).

    Usage::

        evaluator = GameEvaluator()

        for ep in range(num_games):
            game = Game2048({'grid_size': 4})
            evaluator.start_episode()

            while not game.is_game_over():
                t0 = time.time()
                action = agent.choose(game.get_state())
                inference_ms = (time.time() - t0) * 1000
                evaluator.log_move(inference_ms)
                game.move(action)

            evaluator.end_episode(game)

        evaluator.print_summary("MCTS")
    """

    def __init__(self):
        self.episodes: List[EpisodeStats] = []
        self._current: Optional[EpisodeStats] = None
        self.training_time_sec: float = 0.0
        self.training_episodes: int = 0

    # ─── Per-Episode Tracking ─────────────────────────────────────

    def start_episode(self):
        """Call at the start of each game."""
        self._current = EpisodeStats()

    def log_move(self, inference_ms: float):
        """Log inference time for a single move.

        Args:
            inference_ms: Time in milliseconds the agent took to decide.
        """
        if self._current is None:
            raise RuntimeError("Call start_episode() before log_move()")
        self._current.move_times_ms.append(inference_ms)
        self._current.move_count += 1

    def end_episode(self, game) -> EpisodeStats:
        """Finalize metrics for a completed game.

        Extracts final score, highest tile, tile distribution, and
        win conditions from the game state.

        Args:
            game: A Game2048 instance that has finished.

        Returns:
            EpisodeStats for this episode.
        """
        if self._current is None:
            raise RuntimeError("Call start_episode() before end_episode()")

        ep = self._current
        board = game.get_state()

        ep.score = game.get_score()
        ep.highest_tile = int(np.max(board))
        ep.reached_2048 = ep.highest_tile >= 2048
        ep.reached_4096 = ep.highest_tile >= 4096
        ep.reached_8192 = ep.highest_tile >= 8192

        # Tile distribution on final board (non-zero tiles)
        unique, counts = np.unique(board[board > 0], return_counts=True)
        ep.tile_dist = {int(v): int(c) for v, c in zip(unique, counts)}

        self.episodes.append(ep)
        self._current = None
        return ep

    # ─── Training Cost Tracking (for RL agents) ──────────────────

    def set_training_stats(self, training_time_sec: float, training_episodes: int):
        """Record training cost (only applicable for DQN, PPO, AlphaZero).

        Args:
            training_time_sec: Total wall-clock training time in seconds.
            training_episodes: Number of self-play episodes during training.
        """
        self.training_time_sec = training_time_sec
        self.training_episodes = training_episodes

    # ─── Performance Metrics ──────────────────────────────────────

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def win_rate(self, threshold: int = 2048) -> float:
        """Percentage of games that reached the given tile.

        Args:
            threshold: Tile value to count as a win (default 2048).

        Returns:
            float: Win rate as a percentage (0-100).
        """
        if not self.episodes:
            return 0.0
        wins = sum(1 for ep in self.episodes if ep.highest_tile >= threshold)
        return (wins / len(self.episodes)) * 100

    def avg_score(self) -> float:
        """Average merge score across all episodes."""
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.score for ep in self.episodes]))

    def avg_moves(self) -> float:
        """Average number of moves per game (survival length)."""
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.move_count for ep in self.episodes]))

    def highest_tile_distribution(self) -> Dict[int, float]:
        """Distribution of highest tile achieved across all games.

        Returns:
            Dict mapping tile values to percentage of games ending
            with that tile as the highest (e.g. {512: 25.0, 1024: 50.0, 2048: 25.0}).
        """
        if not self.episodes:
            return {}
        tiles = [ep.highest_tile for ep in self.episodes]
        unique, counts = np.unique(tiles, return_counts=True)
        total = len(tiles)
        return {int(v): round(float(c / total * 100), 2) for v, c in zip(unique, counts)}

    # ─── Efficiency Metrics ───────────────────────────────────────

    def avg_inference_time_ms(self) -> float:
        """Mean inference time per move in milliseconds."""
        all_times = []
        for ep in self.episodes:
            all_times.extend(ep.move_times_ms)
        if not all_times:
            return 0.0
        return float(np.mean(all_times))

    def p95_inference_time_ms(self) -> float:
        """95th percentile inference time per move in milliseconds."""
        all_times = []
        for ep in self.episodes:
            all_times.extend(ep.move_times_ms)
        if not all_times:
            return 0.0
        return float(np.percentile(all_times, 95))

    def score_per_compute_second(self) -> float:
        """Composite efficiency: avg_score / total_wall_clock_seconds.

        Includes both training time and evaluation time. For search agents
        with no training, this is purely evaluation cost.

        Returns:
            float: Score per second of compute. Higher = more efficient.
        """
        all_times = []
        for ep in self.episodes:
            all_times.extend(ep.move_times_ms)

        eval_time_sec = sum(all_times) / 1000.0 if all_times else 0.0
        total_sec = self.training_time_sec + eval_time_sec

        if total_sec == 0:
            return 0.0
        return self.avg_score() / total_sec

    # ─── Summary & Export ─────────────────────────────────────────

    def get_summary(self, agent_name: str = "Agent") -> Dict:
        """Generate a complete summary dict of all metrics.

        Args:
            agent_name: Name/label for this agent configuration.

        Returns:
            Dict containing all performance and efficiency metrics.
        """
        return {
            'agent': agent_name,
            'num_episodes': self.num_episodes,
            # Performance
            'win_rate_2048': round(self.win_rate(2048), 2),
            'win_rate_4096': round(self.win_rate(4096), 2),
            'win_rate_8192': round(self.win_rate(8192), 2),
            'avg_merge_score': round(self.avg_score(), 2),
            'avg_moves_per_game': round(self.avg_moves(), 2),
            'highest_tile_distribution': self.highest_tile_distribution(),
            # Efficiency
            'avg_inference_ms': round(self.avg_inference_time_ms(), 3),
            'p95_inference_ms': round(self.p95_inference_time_ms(), 3),
            'training_time_sec': round(self.training_time_sec, 2),
            'training_episodes': self.training_episodes,
            'score_per_compute_sec': round(self.score_per_compute_second(), 2),
        }

    def print_summary(self, agent_name: str = "Agent"):
        """Pretty-print evaluation results to stdout.

        Args:
            agent_name: Name/label for this agent.
        """
        s = self.get_summary(agent_name)

        print(f"\n{'=' * 60}")
        print(f"  {agent_name} — Evaluation Summary ({s['num_episodes']} games)")
        print(f"{'=' * 60}")

        print(f"\n  PERFORMANCE METRICS")
        print(f"  {'─' * 40}")
        print(f"  Win Rate (≥2048):      {s['win_rate_2048']}%")
        print(f"  Win Rate (≥4096):      {s['win_rate_4096']}%")
        print(f"  Win Rate (≥8192):      {s['win_rate_8192']}%")
        print(f"  Avg Merge Score:       {s['avg_merge_score']}")
        print(f"  Avg Moves/Game:        {s['avg_moves_per_game']}")
        print(f"  Highest Tile Dist:     {s['highest_tile_distribution']}")

        print(f"\n  EFFICIENCY METRICS")
        print(f"  {'─' * 40}")
        print(f"  Avg Inference:         {s['avg_inference_ms']} ms/move")
        print(f"  95th %ile Inference:   {s['p95_inference_ms']} ms/move")
        if s['training_time_sec'] > 0:
            print(f"  Training Time:         {s['training_time_sec']}s "
                  f"({s['training_episodes']} episodes)")
        else:
            print(f"  Training Time:         N/A (search agent)")
        print(f"  Score/Compute-sec:     {s['score_per_compute_sec']}")
        print(f"{'=' * 60}\n")

    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file.

        Args:
            filepath: Output file path (e.g. 'mcts_results.json').
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"Results saved to {filepath}")

    def reset(self):
        """Clear all collected data for a fresh evaluation run."""
        self.episodes.clear()
        self._current = None
        self.training_time_sec = 0.0
        self.training_episodes = 0


# ═══════════════════════════════════════════════════════════════════
#  MULTI-AGENT COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_agents(results: List[Dict]) -> None:
    """Print a side-by-side comparison table of multiple agents.

    Args:
        results: List of summary dicts from GameEvaluator.get_summary().

    Example::

        mcts_summary = mcts_evaluator.get_summary("MCTS")
        dqn_summary  = dqn_evaluator.get_summary("DQN")
        compare_agents([mcts_summary, dqn_summary])
    """
    if not results:
        print("No results to compare.")
        return

    agents = [r['agent'] for r in results]
    col_w = max(20, max(len(a) for a in agents) + 2)

    header = f"{'Metric':<28}" + "".join(f"{a:>{col_w}}" for a in agents)
    print(f"\n{'═' * len(header)}")
    print(header)
    print(f"{'═' * len(header)}")

    metrics = [
        ('Win Rate ≥2048 (%)',   'win_rate_2048'),
        ('Win Rate ≥4096 (%)',   'win_rate_4096'),
        ('Avg Merge Score',      'avg_merge_score'),
        ('Avg Moves/Game',       'avg_moves_per_game'),
        ('Avg Inference (ms)',   'avg_inference_ms'),
        ('P95 Inference (ms)',   'p95_inference_ms'),
        ('Training Time (s)',    'training_time_sec'),
        ('Score/Compute-sec',    'score_per_compute_sec'),
    ]

    for label, key in metrics:
        row = f"{label:<28}"
        for r in results:
            row += f"{r.get(key, 'N/A'):>{col_w}}"
        print(row)

    print(f"{'═' * len(header)}\n")


# ═══════════════════════════════════════════════════════════════════
#  DEMO — run with: python evaluation.py
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from game import Game2048, Action

    print("Running evaluation demo with random agent (20 games)...\n")

    evaluator = GameEvaluator()
    reward_fn = REWARD_SEARCH

    for ep in range(20):
        game = Game2048({'grid_size': 4})
        evaluator.start_episode()

        while not game.is_game_over():
            available = game.get_available_moves()
            if not available:
                break

            t0 = time.time()
            action = available[np.random.randint(len(available))]
            inference_ms = (time.time() - t0) * 1000

            evaluator.log_move(inference_ms)
            game.move(action)

        stats = evaluator.end_episode(game)

        # Show reward breakdown for final board
        if ep == 0:
            print("Reward breakdown for game 1 final board:")
            breakdown = reward_fn.compute_breakdown(game.get_state())
            for k, v in breakdown.items():
                print(f"  {k:<20} {v:.4f}")
            print()

    evaluator.print_summary("Random Agent")