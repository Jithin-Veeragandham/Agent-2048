"""
logger.py
========

Logging utilities for 2048 agent experiments.

Provides ``RunLogger``, a lightweight logger that the InteractionModule
calls at each move and episode boundary.

**Log architecture (v2):**

Single unified log file — ALL agents, ALL params go here:

    logs/all_runs.jsonl   — one JSON line per GAME (not per episode)

Each line contains full context so you can filter/group/aggregate:

    run_id          — UUID shared by all games in one run() call
    game_index      — 0-based index within this run (0, 1, 2, ...)
    num_games_in_run— total games in this run (so you see "game 3 of 10")
    agent           — agent name string
    agent_params    — dict of architecture-specific params
    config          — game config (grid_size, seed, etc.)
    reward_weights  — reward function weights used
    timestamp       — ISO timestamp when game finished
    score           — final merge score
    highest_tile    — max tile on final board
    moves           — total move count
    reached_2048    — bool
    reached_4096    — bool
    won             — alias for reached_2048
    time_sec        — wall-clock seconds for this game
    avg_inference_ms— mean agent decision time per move
    p50_inference_ms— median inference time
    p95_inference_ms— 95th percentile inference time
    empty_cells     — empty cells on final board
    avg_reward_breakdown         — per-move average of heuristic components
    final_reward_breakdown       — heuristic breakdown of the final board
    quartile_reward_breakdowns   — breakdowns sampled at Q25/Q50/Q75/Q100 of
                                   the move sequence, for trajectory analysis
    final_board     — final board as nested list

The latest run's full per-move detail is saved to:

    logs/latest_run.json

This lets you:
    - Filter by agent_params to compare beam_width=10 vs 15
    - Group by run_id to find all games in one episode
    - Compute win rates: sum(won) / count where agent_params match
    - Compare architectures: group by agent, aggregate scores

Usage::

    from framework.logger import RunLogger
    from framework.interaction import InteractionModule
    from agents.beam_search import BeamSearchAgent

    agent = BeamSearchAgent(beam_width=10, search_depth=15)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=10)
    # logs/all_runs.jsonl now has 10 lines (one per game)
    # logs/latest_run.json has full per-move detail

Reading logs back::

    from framework.logger import load_all_runs, load_latest_run

    # All historical games (every agent, every config)
    all_games = load_all_runs()

    # Filter to specific params
    beam_10 = [g for g in all_games
               if g['agent_params'].get('beam_width') == 10]

    # Win rate for a specific config
    wins = sum(1 for g in beam_10 if g['won'])
    print(f"Win rate: {wins}/{len(beam_10)} = {wins/len(beam_10)*100:.1f}%")

    # Group by run_id to see episode-level stats
    from itertools import groupby
    by_run = groupby(sorted(all_games, key=lambda g: g['run_id']),
                     key=lambda g: g['run_id'])
"""

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
        log_dir:         Directory for log files (created on save if missing).
        log_move_detail: If True, include the full per-move reward breakdown
                         list in every game record (``move_reward_breakdowns``).
                         Off by default — files are ~400x larger when enabled.
    """

    def __init__(self, log_dir: str = "logs", log_move_detail: bool = False):
        self.log_dir = log_dir
        self.log_move_detail = log_move_detail

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
        quartile_reward_breakdowns: Optional[Dict] = None,
        move_reward_breakdowns: Optional[List[Dict]] = None,
    ):
        """Finalize a game record.

        Args:
            final_score:                 Final cumulative score.
            highest_tile:                Max tile on the final board.
            move_count:                  Total moves in this game.
            reached_2048:                Whether 2048 tile was achieved.
            final_board:                 Final board state.
            final_reward_breakdown:      Breakdown of the final board.
            quartile_reward_breakdowns:  Pre-computed quartile snapshots
                                         (used by parallel path). If None,
                                         computed from self._current_moves.
            move_reward_breakdowns:  Full per-move breakdown list (used by
                                         parallel path when log_move_detail=True).
                                         If None, extracted from self._current_moves.
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
        move_breakdowns = [m['reward_breakdown'] for m in self._current_moves]
        avg_breakdown = _average_breakdowns(move_breakdowns)

        # Quartile snapshots (Q25/Q50/Q75/Q100) — or use pre-computed from parallel path
        if quartile_reward_breakdowns is not None:
            quartiles = quartile_reward_breakdowns
        else:
            quartiles = _quartile_breakdowns(move_breakdowns)

        # Full per-move list — only populated when log_move_detail=True
        if self.log_move_detail:
            if move_reward_breakdowns is not None:
                per_move = move_reward_breakdowns  # parallel path
            else:
                per_move = move_breakdowns         # sequential path
        else:
            per_move = None

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
            'avg_reward_breakdown':         avg_breakdown,
            'final_reward_breakdown':       {k: round(v, 6) for k, v in final_reward_breakdown.items()},
            'quartile_reward_breakdowns':   quartiles,
            **({'move_reward_breakdowns': per_move} if per_move is not None else {}),

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


def _quartile_breakdowns(breakdowns: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Sample reward breakdowns at Q25, Q50, Q75, Q100 of the move sequence.

    Returns a dict like:
        {
            "q25":  {"tile_score": ..., "smoothness": ..., ...},
            "q50":  {...},
            "q75":  {...},
            "q100": {...},   # same board as final_reward_breakdown
        }

    Use consecutive deltas (q50 - q25, etc.) to see what each agent
    is optimising for across the arc of the game.
    """
    n = len(breakdowns)
    if n == 0:
        return {}
    indices = {
        'q25':  int(n * 0.25),
        'q50':  int(n * 0.50),
        'q75':  int(n * 0.75),
        'q100': n - 1,
    }
    # clamp to valid range
    indices = {label: min(max(idx, 0), n - 1) for label, idx in indices.items()}
    return {
        label: {k: round(float(v), 6) for k, v in breakdowns[idx].items()}
        for label, idx in indices.items()
    }
