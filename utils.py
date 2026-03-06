"""
utils.py
========

Logging utilities for 2048 agent experiments.

Provides ``RunLogger``, a lightweight logger that the InteractionModule
calls at each move and episode boundary. Produces two log files per
algorithm:

    logs/<algo>_latest_run.json    — full detail for the most recent run
    logs/<algo>_all_runs.jsonl     — one summary line appended per run

Latest run file contains:
    - Per-move snapshots: board state, action, reward, score,
      reward function breakdown, inference time
    - Run metadata: timestamp, config, total steps, final metrics

All runs file contains (one JSON line per run, appended):
    - Timestamp
    - Final metrics: score, highest tile, move count, total time
    - Per-move average reward breakdown
    - Final board reward breakdown

Usage::

    from utils import RunLogger
    from interaction import InteractionModule
    from beam_search import BeamSearchAgent
    from evaluation import REWARD_SEARCH

    agent = BeamSearchAgent(beam_width=15, search_depth=20)
    logger = RunLogger()

    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        reward_fn=REWARD_SEARCH,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=10)
    module.print_results()
    # logs written automatically

Reading logs back::

    from utils import load_all_runs, load_latest_run

    runs = load_all_runs("BeamSearch(w=15,d=20)")
    latest = load_latest_run("BeamSearch(w=15,d=20)")
"""

import os
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

from game import Action
from evaluation import RewardFunction


# ═══════════════════════════════════════════════════════════════════
#  RUN LOGGER
# ═══════════════════════════════════════════════════════════════════

class RunLogger:
    """Lightweight logger called by InteractionModule at each move.

    The logger does not run any game logic — it only records data
    that the interaction module passes to it. Three call sites:

        logger.log_move(...)    — after each move
        logger.end_episode(...) — after each game ends
        logger.save(...)        — after all games in a run complete

    Args:
        log_dir: Directory for log files (created on save if missing).

    Attributes:
        episode_logs: List of per-episode records accumulated during a run.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir

        # Accumulated across the full run
        self._episode_logs: List[Dict] = []

        # Current episode state
        self._current_moves: List[Dict] = []
        self._run_start: float = 0.0

    # ─── Called by InteractionModule ──────────────────────────

    def on_run_start(self):
        """Called once at the beginning of a run() call."""
        self._episode_logs = []
        self._run_start = time.time()

    def on_episode_start(self):
        """Called at the start of each episode."""
        self._current_moves = []

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
            step:             Move number within this episode.
            state:            Board state BEFORE the move.
            action:           Action taken.
            reward:           Merge reward from this move.
            score:            Cumulative score AFTER the move.
            inference_ms:     Agent decision time in milliseconds.
            reward_breakdown: Dict from RewardFunction.compute_breakdown().
        """
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
        """Finalize an episode record.

        Args:
            final_score:            Final cumulative score.
            highest_tile:           Max tile on the final board.
            move_count:             Total moves in this episode.
            reached_2048:           Whether 2048 tile was achieved.
            final_board:            Final board state.
            final_reward_breakdown: Breakdown of the final board.
        """
        avg_breakdown = _average_breakdowns(
            [m['reward_breakdown'] for m in self._current_moves]
        )

        self._episode_logs.append({
            'episode': len(self._episode_logs) + 1,
            'score': int(final_score),
            'highest_tile': int(highest_tile),
            'moves': int(move_count),
            'reached_2048': reached_2048,
            'avg_reward_breakdown': avg_breakdown,
            'final_reward_breakdown': {k: round(v, 6) for k, v in final_reward_breakdown.items()},
            'final_board': final_board.tolist(),
            'move_log': self._current_moves,
        })
        self._current_moves = []

    def save(self, agent_name: str, config: Dict):
        """Write both log files. Called by InteractionModule after run().

        Args:
            agent_name: Agent's name (used in filenames).
            config:     Game config dict (stored in logs for reproducibility).
        """
        if not self._episode_logs:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        algo = _safe_filename(agent_name)
        total_time = time.time() - self._run_start
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Latest run (full detail) ──────────────────────
        latest_path = os.path.join(self.log_dir, f"{algo}_latest_run.json")
        latest = {
            'timestamp': timestamp,
            'agent': agent_name,
            'config': config,
            'num_episodes': len(self._episode_logs),
            'total_time_sec': round(total_time, 3),
            'episodes': self._episode_logs,
        }
        with open(latest_path, 'w') as f:
            json.dump(latest, f, indent=2)
        print(f"Latest run log saved to {latest_path}")

        # ── All runs (append summary line) ────────────────
        all_path = os.path.join(self.log_dir, f"{algo}_all_runs.jsonl")

        scores = [ep['score'] for ep in self._episode_logs]
        highest_tiles = [ep['highest_tile'] for ep in self._episode_logs]
        move_counts = [ep['moves'] for ep in self._episode_logs]

        summary_line = {
            'timestamp': timestamp,
            'agent': agent_name,
            'config': {'grid_size': config.get('grid_size', 4)},
            'num_episodes': len(self._episode_logs),
            'total_time_sec': round(total_time, 3),
            'metrics': {
                'avg_score': round(float(np.mean(scores)), 2),
                'max_score': int(np.max(scores)),
                'min_score': int(np.min(scores)),
                'avg_highest_tile': round(float(np.mean(highest_tiles)), 2),
                'max_highest_tile': int(np.max(highest_tiles)),
                'avg_moves': round(float(np.mean(move_counts)), 2),
                'total_moves': int(np.sum(move_counts)),
            },
            'avg_reward_breakdown_per_move': _average_breakdowns(
                [ep['avg_reward_breakdown'] for ep in self._episode_logs]
            ),
            'avg_final_reward_breakdown': _average_breakdowns(
                [ep['final_reward_breakdown'] for ep in self._episode_logs]
            ),
        }
        with open(all_path, 'a') as f:
            f.write(json.dumps(summary_line) + '\n')
        print(f"Run summary appended to {all_path}")


# ═══════════════════════════════════════════════════════════════════
#  LOG READERS
# ═══════════════════════════════════════════════════════════════════

def load_all_runs(agent_name: str, log_dir: str = "logs") -> List[Dict]:
    """Load all historical run summaries for an agent.

    Args:
        agent_name: Agent name (e.g. "BeamSearch(w=15,d=20)").
        log_dir:    Directory containing log files.

    Returns:
        List of summary dicts, one per past run, chronological order.

    Example::

        from utils import load_all_runs
        runs = load_all_runs("BeamSearch(w=15,d=20)")
        for r in runs:
            print(r['timestamp'], r['metrics']['avg_score'])
    """
    path = os.path.join(log_dir, f"{_safe_filename(agent_name)}_all_runs.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_latest_run(agent_name: str, log_dir: str = "logs") -> Optional[Dict]:
    """Load the full latest run log for an agent.

    Args:
        agent_name: Agent name (e.g. "BeamSearch(w=15,d=20)").
        log_dir:    Directory containing log files.

    Returns:
        Dict with full episode detail, or None if no log exists.

    Example::

        from utils import load_latest_run
        run = load_latest_run("BeamSearch(w=15,d=20)")
        if run:
            for ep in run['episodes']:
                print(f"Episode {ep['episode']}: score={ep['score']}")
    """
    path = os.path.join(log_dir, f"{_safe_filename(agent_name)}_latest_run.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


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


def _safe_filename(name: str) -> str:
    """Convert an agent name to a filesystem-safe string.

    "BeamSearch(w=15,d=20)" -> "BeamSearch_w_15_d_20"
    """
    for ch in "()=,/ \t":
        name = name.replace(ch, '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')