import os
import json
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

from game import Action


# ═══════════════════════════════════════════════════════════════════
#  RUN LOGGER
# ═══════════════════════════════════════════════════════════════════

class RunLogger:
    """Lightweight logger called by InteractionModule at each move.

    Produces one JSON line per game in a single unified log file.
    All agent types, all parameter configs go to the same file —
    architecture info is stored as fields, not in filenames.

    Call sites (from InteractionModule):
        logger.on_run_start(num_games, agent_name, agent_params, config, reward_weights)
        logger.on_episode_start()
        logger.log_move(...)
        logger.end_episode(...)
        logger.save()

    Args:
        log_dir: Directory for log files (created on save if missing).
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir

        # ── Run-level state ───────────────────────────────────
        self._run_id: str = ""
        self._num_games: int = 0
        self._game_index: int = 0
        self._agent_name: str = ""
        self._agent_params: Dict = {}
        self._config: Dict = {}
        self._reward_weights: Dict = {}

        # ── Accumulated per-game records for this run ─────────
        self._game_records: List[Dict] = []

        # ── Current game state ────────────────────────────────
        self._current_moves: List[Dict] = []
        self._current_inference_times: List[float] = []
        self._game_start: float = 0.0

    # ─── Called by InteractionModule ──────────────────────────

    def on_run_start(
        self,
        num_games: int,
        agent_name: str,
        agent_type: str,
        agent_params: Dict,
        config: Dict,
        reward_weights: Dict,
    ):
        """Called once at the beginning of a run() call.

        Args:
            num_games:      Total games in this run.
            agent_name:     Agent's display name.
            agent_type:     Agent type slug for filename (e.g. "beam_search").
            agent_params:   Architecture-specific params (beam_width, etc.).
            config:         Game config dict.
            reward_weights: Reward function weight dict.
        """
        self._run_id = str(uuid.uuid4())[:12]
        self._num_games = num_games
        self._game_index = 0
        self._agent_name = agent_name
        self._agent_type = agent_type
        self._agent_params = agent_params
        self._config = config
        self._reward_weights = reward_weights
        self._game_records = []

    def on_episode_start(self):
        """Called at the start of each game."""
        self._current_moves = []
        self._current_inference_times = []
        self._game_start = time.time()

    def log_move(
        self,
        step: int,
        state: np.ndarray,
        action: Action,
        reward: int,
        score: int,
        inference_ms: float,
        reward_breakdown: Dict[str, float],
    ):
        """Record one move's data.

        Args:
            step:             Move number within this game.
            state:            Board state BEFORE the move.
            action:           Action taken.
            reward:           Merge reward from this move.
            score:            Cumulative score AFTER the move.
            inference_ms:     Agent decision time in milliseconds.
            reward_breakdown: Dict from RewardFunction.compute_breakdown().
        """
        self._current_inference_times.append(inference_ms)
        self._current_moves.append({
            'step': step,
            'action': action.name,
            'reward': int(reward),
            'score': int(score),
            'inference_ms': round(inference_ms, 3),
            'board_state': state.tolist(),
            'reward_breakdown': {k: round(v, 6) for k, v in reward_breakdown.items()},
        })

    def end_episode(
        self,
        final_score: int,
        highest_tile: int,
        move_count: int,
        reached_2048: bool,
        final_board: np.ndarray,
        final_reward_breakdown: Dict[str, float],
    ):
        """Finalize a game record.

        Args:
            final_score:            Final cumulative score.
            highest_tile:           Max tile on the final board.
            move_count:             Total moves in this game.
            reached_2048:           Whether 2048 tile was achieved.
            final_board:            Final board state.
            final_reward_breakdown: Breakdown of the final board.
        """
        game_time = time.time() - self._game_start
        inf_times = self._current_inference_times

        # Inference time stats
        if inf_times:
            avg_inf = float(np.mean(inf_times))
            p50_inf = float(np.median(inf_times))
            p95_inf = float(np.percentile(inf_times, 95))
        else:
            avg_inf = p50_inf = p95_inf = 0.0

        # Per-move average reward breakdown
        avg_breakdown = _average_breakdowns(
            [m['reward_breakdown'] for m in self._current_moves]
        )

        # Empty cells on final board
        final_arr = final_board if isinstance(final_board, np.ndarray) else np.array(final_board)
        empty_cells = int(np.sum(final_arr == 0))

        # Tile distribution on final board
        non_zero = final_arr[final_arr > 0]
        unique, counts = np.unique(non_zero, return_counts=True)
        tile_dist = {int(v): int(c) for v, c in zip(unique, counts)}

        record = {
            # ── Identity / grouping ───────────────────────
            'run_id':             self._run_id,
            'game_index':         self._game_index,
            'num_games_in_run':   self._num_games,
            'timestamp':          datetime.now(timezone.utc).isoformat(),

            # ── Architecture / config ─────────────────────
            'agent':              self._agent_name,
            'agent_params':       self._agent_params,
            'config':             self._config,
            'reward_weights':     self._reward_weights,

            # ── Game outcome metrics ──────────────────────
            'score':              int(final_score),
            'highest_tile':       int(highest_tile),
            'moves':              int(move_count),
            'reached_2048':       reached_2048,
            'reached_4096':       highest_tile >= 4096,
            'won':                reached_2048,

            # ── Timing ────────────────────────────────────
            'time_sec':           round(game_time, 3),
            'avg_inference_ms':   round(avg_inf, 3),
            'p50_inference_ms':   round(p50_inf, 3),
            'p95_inference_ms':   round(p95_inf, 3),

            # ── Board analysis ────────────────────────────
            'empty_cells':        empty_cells,
            'tile_distribution':  tile_dist,

            # ── Reward breakdowns ─────────────────────────
            'avg_reward_breakdown':   avg_breakdown,
            'final_reward_breakdown': {k: round(v, 6) for k, v in final_reward_breakdown.items()},

            # ── Final board ───────────────────────────────
            'final_board':        final_arr.tolist(),
        }

        self._game_records.append(record)
        self._game_index += 1
        self._current_moves = []
        self._current_inference_times = []

        # ── Write immediately to jsonl ────────────────────
        os.makedirs(self.log_dir, exist_ok=True)
        log_filename = f"{self._agent_type}_runs.jsonl"
        log_path = os.path.join(self.log_dir, log_filename)
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def save(self):
        """Write latest_run.json with full per-move detail.

        Called by InteractionModule after run(). The per-game jsonl
        lines are already written in end_episode(), so this only
        handles the detailed latest run file.
        """
        if not self._game_records:
            return

        os.makedirs(self.log_dir, exist_ok=True)

        # ── Latest run with full per-move detail ──────────
        latest_path = os.path.join(self.log_dir, "latest_run.json")

        latest = {
            'run_id':        self._run_id,
            'timestamp':     datetime.now(timezone.utc).isoformat(),
            'agent':         self._agent_name,
            'agent_type':    self._agent_type,
            'agent_params':  self._agent_params,
            'config':        self._config,
            'reward_weights': self._reward_weights,
            'num_games':     len(self._game_records),
            'games':         self._game_records,
        }

        with open(latest_path, 'w') as f:
            json.dump(latest, f, indent=2)
        print(f"  Latest run detail saved to {latest_path}")


# ═══════════════════════════════════════════════════════════════════
#  LOG READERS
# ═══════════════════════════════════════════════════════════════════

def load_all_runs(agent_type: str, log_dir: str = "logs") -> List[Dict]:
    """Load all historical game records for an agent type.

    Args:
        agent_type: Agent type slug (e.g. "beam_search", "mcts").
        log_dir:    Directory containing log files.

    Returns:
        List of dicts, one per game, chronological order.

    Example::

        games = load_all_runs("beam_search")

        # Filter to beam search w=10
        bs10 = [g for g in games
                if g['agent_params'].get('beam_width') == 10]

        # Win rate
        wins = sum(1 for g in bs10 if g['won'])
        print(f"{wins}/{len(bs10)} = {wins/len(bs10)*100:.1f}%")
    """
    path = os.path.join(log_dir, f"{agent_type}_runs.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_latest_run(log_dir: str = "logs") -> Optional[Dict]:
    """Load the full latest run detail (includes per-move logs).

    Returns:
        Dict with full run detail, or None if no log exists.
    """
    path = os.path.join(log_dir, "latest_run.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_runs_by_agent(agent_name: str, log_dir: str = "logs") -> List[Dict]:
    """Load all games for a specific agent name.

    Args:
        agent_name: Exact agent name string (e.g. "BeamSearch(w=10,d=15)").

    Returns:
        Filtered list of game records.
    """
    return [g for g in load_all_runs(log_dir) if g['agent'] == agent_name]


def load_runs_by_params(log_dir: str = "logs", **params) -> List[Dict]:
    """Load all games matching specific agent_params.

    Args:
        **params: Key-value pairs to match in agent_params.

    Returns:
        Filtered list of game records.

    Example::

        # All beam search games with width=10
        games = load_runs_by_params(beam_width=10)

        # All MCTS games with 500 simulations
        games = load_runs_by_params(num_simulations=500)
    """
    all_games = load_all_runs(log_dir)
    filtered = []
    for g in all_games:
        ap = g.get('agent_params', {})
        if all(ap.get(k) == v for k, v in params.items()):
            filtered.append(g)
    return filtered


def summarize_games(games: List[Dict]) -> Dict:
    """Compute summary statistics for a list of game records.

    Args:
        games: List of game record dicts (from load_all_runs, etc.).

    Returns:
        Dict with aggregated stats.

    Example::

        games = load_runs_by_params(beam_width=10, search_depth=15)
        summary = summarize_games(games)
        print(f"Win rate: {summary['win_rate_2048']}%")
    """
    if not games:
        return {}

    scores = [g['score'] for g in games]
    tiles = [g['highest_tile'] for g in games]
    moves = [g['moves'] for g in games]
    times = [g['time_sec'] for g in games]
    wins_2048 = sum(1 for g in games if g.get('won', False))
    wins_4096 = sum(1 for g in games if g.get('reached_4096', False))
    n = len(games)

    # Tile distribution
    unique_tiles, counts = np.unique(tiles, return_counts=True)
    tile_dist = {
        int(t): f"{int(c)}/{n} ({c/n*100:.1f}%)"
        for t, c in zip(unique_tiles, counts)
    }

    # Unique run IDs
    run_ids = set(g['run_id'] for g in games)

    return {
        'num_games':       n,
        'num_runs':        len(run_ids),
        'win_rate_2048':   round(wins_2048 / n * 100, 2),
        'win_rate_4096':   round(wins_4096 / n * 100, 2),
        'avg_score':       round(float(np.mean(scores)), 2),
        'max_score':       int(np.max(scores)),
        'min_score':       int(np.min(scores)),
        'std_score':       round(float(np.std(scores)), 2),
        'avg_highest_tile': round(float(np.mean(tiles)), 2),
        'max_highest_tile': int(np.max(tiles)),
        'avg_moves':       round(float(np.mean(moves)), 2),
        'avg_time_sec':    round(float(np.mean(times)), 3),
        'total_time_sec':  round(float(np.sum(times)), 3),
        'tile_distribution': tile_dist,
    }


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _average_breakdowns(breakdowns: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of reward breakdown dicts, key by key."""
    if not breakdowns:
        return {}
    keys = breakdowns[0].keys()
    return {
        k: round(float(np.mean([b[k] for b in breakdowns if k in b])), 6)
        for k in keys
    }