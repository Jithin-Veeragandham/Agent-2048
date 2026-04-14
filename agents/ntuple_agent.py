# coding: utf-8
"""
agents/ntuple_agent.py
======================
Fully self-contained N-Tuple TD agent -- no external dependencies.
All required game logic from abachurin/2048 is embedded directly in
this file so it runs on any machine without PYTHONPATH setup.

Architecture
------------
N-tuple TD(0) learning with a linear value function over hand-crafted
board features (rows, columns, 2x2 squares). The agent uses a greedy
policy -- no exploration at inference time. Action selection works by
trying all 4 legal moves, scoring each resulting board state with the
learned weight table, and picking the highest-scoring one.

Board representation:
    Agent-2048 (Game2048): stores actual tile values (0, 2, 4, 8, ...)
    QAgent (abachurin):    stores log2 values (0=empty, 1=2, 2=4, ...)
    This wrapper converts between the two at every step.

Usage
-----
    from agents.ntuple_agent import NTupleAgent
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    agent  = NTupleAgent(agent_path="checkpoints/ntuple_best_agent.pkl",
                         weights_path="checkpoints/ntuple_best_agent_weights.pkl")
    logger = RunLogger(log_move_detail=True)
    module = InteractionModule(
        config={"grid_size": 4},
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=100)
    module.print_results()

Quick eval from command line
----------------------------
    python agents/ntuple_agent.py \
        --agent checkpoints/ntuple_best_agent.pkl \
        --weights checkpoints/ntuple_best_agent_weights.pkl \
        --games 100
"""

import os
import sys
import json
import pickle
import random
import numpy as np
from typing import Dict, List, Optional, Any

# Ensure repo root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from game import Action
from agents.base import BaseAgent


# ================================================================
#  N-TUPLE FEATURE EXTRACTION  (from abachurin/2048 r_learning.py)
#  Copied here so no external package is needed at runtime.
#
#  f_6 returns 33 feature indices, one per weight table:
#    indices  0-16  → 17 four-tile features  (16^4 = 65536 entries)
#    indices 17-20  →  4 five-tile features  (16^5 = 1048576 entries)
#    indices 21-32  → 12 six-tile features   (14^6 = 7529536 entries,
#                                             tiles capped at 13)
# ================================================================

def _f_6(x: np.ndarray) -> np.ndarray:
    """N-tuple feature extraction matching game2048.r_learning.f_6.

    Args:
        x: (4,4) int32 board in log2 representation (0=empty, 1=tile-2 …)

    Returns:
        numpy array of 33 integer feature indices.
    """
    # 4-tile column features  (4 columns → 4 indices, range 0..16^4-1)
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    # 4-tile row features  (4 rows → 4 indices)
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    # 4-tile 2×2 square features  (3×3 sliding window → 9 indices)
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    # 5-tile cross features centred on 2×2 interior  (→ 4 indices, range 0..16^5-1)
    x_middle = (
        (x[1:3, 1:3] << 16) + (x[:2, 1:3] << 12) +
        (x[1:3, :2]  << 8)  + (x[2:, 1:3] << 4) + x[1:3, 2:]
    ).ravel()
    # 6-tile vertical-block features — tiles capped at 13, base-14 encoding
    # (3 col × 2 row sliding blocks → 6 indices, range 0..14^6-1)
    y = np.minimum(x, 13)
    x_vert_6 = (
        537824 * y[0:2, 0:3] + 38416 * y[1:3, 0:3] + 2744 * y[2:, 0:3] +
           196 * y[0:2, 1:]  +    14 * y[1:3, 1:]  +       y[2:, 1:]
    ).ravel()
    # 6-tile horizontal-block features  (→ 6 indices)
    x_hor_6 = (
        537824 * y[0:3, 0:2] + 38416 * y[0:3, 1:3] + 2744 * y[0:3, 2:] +
           196 * y[1:, 0:2]  +    14 * y[1:, 1:3]  +       y[1:, 2:]
    ).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle, x_vert_6, x_hor_6])


# ================================================================
#  EMBEDDED GAME LOGIC (from abachurin/2048)
#  Included here so no external repo or PYTHONPATH is needed.
# ================================================================

def _create_move_table():
    """Precompute all possible row-left-slide results.

    For each possible 4-tile row (16^4 = 65536 combinations),
    stores (result_row, score_gained, did_change).
    Uses log2 tile values: 0=empty, 1=tile-2, 2=tile-4, etc.
    """
    table = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    score = 0
                    line = (a, b, c, d)
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, False)
                        continue
                    line_1 = [v for v in line if v]
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)
                            line_1[i], line_1[i + 1] = x + 1, 0
                    line_2 = [v for v in line_1 if v]
                    line_2 = tuple(line_2 + [0] * (4 - len(line_2)))
                    table[line] = (line_2, score, line != line_2)
    return table


# Build move table once at import time
_MOVE_TABLE = _create_move_table()


def _pre_move(row: np.ndarray, score: int, direction: int):
    """Simulate a move on a log2 board without modifying it.

    Uses the precomputed move table for fast row sliding.

    Args:
        row:       (4,4) int32 board in log2 representation.
        score:     Current game score.
        direction: 0=left, 1=up, 2=right, 3=down

    Returns:
        (new_row, new_score, changed)
    """
    new_row = np.rot90(row, direction) if direction else row.copy()
    if not direction:
        new_row = row.copy()
    change = False
    new_score = score
    result = new_row.copy()
    for i in range(4):
        line, sc, changed = _MOVE_TABLE[tuple(new_row[i])]
        if changed:
            change = True
            new_score += sc
            result[i] = line
    if direction:
        result = np.rot90(result, 4 - direction)
    return result, new_score, change


# ================================================================
#  ACTION MAPPING
# ================================================================

# QAgent direction: 0=left, 1=up, 2=right, 3=down
# Agent-2048 Action enum: UP=0, DOWN=1, LEFT=2, RIGHT=3
_ACTION_TO_DIR = {
    Action.LEFT:  0,
    Action.UP:    1,
    Action.RIGHT: 2,
    Action.DOWN:  3,
}


# ================================================================
#  BOARD CONVERSION
# ================================================================

def _actual_to_log2(board: np.ndarray) -> np.ndarray:
    """Convert Agent-2048 board (actual tile values) to log2 format.

    Agent-2048: 0=empty, 2=tile-2, 4=tile-4, ...
    QAgent:     0=empty, 1=tile-2, 2=tile-4, ...
    """
    result = np.zeros_like(board, dtype=np.int32)
    nonzero = board > 0
    result[nonzero] = np.log2(board[nonzero]).astype(np.int32)
    return result


# ================================================================
#  NTUPLE AGENT
# ================================================================

class NTupleAgent(BaseAgent):
    """Tabular QAgent (n-tuple TD learning) wrapped as a BaseAgent.

    Fully self-contained -- all game logic is embedded in this file.
    No external repo or PYTHONPATH setup needed on any machine.

    Loads a trained checkpoint and runs greedy inference using the
    learned value function. No neural network -- value is computed as
    a sum of lookup table entries indexed by tile pattern features.

    Args:
        agent_path:   Path to agent .pkl (structure without weights).
        weights_path: Path to weights .pkl (separate file).
        name:         Display name for logs.
    """

    agent_type: str = "ntuple_agent"

    def __reduce__(self):
        """Tell multiprocessing to reconstruct from paths, not the loaded agent."""
        return (NTupleAgent, (self.agent_path, self.weights_path, self.name))

    def __init__(
        self,
        agent_path: str,
        weights_path: Optional[str] = None,
        name: str = "N-Tuple TD Agent",
    ):
        super().__init__(name)
        self.agent_path   = agent_path
        self.weights_path = weights_path

        # Load agent shell and weights
        # The pkl was saved with game2048 module -- we patch sys.modules
        # so pickle can deserialize without the external repo.
        self._agent = self._load_agent(agent_path, weights_path)

        self._n           = self._agent.n
        self._num_feat    = self._agent.num_feat
        self._features    = self._agent.features
        self._trained_eps = self._agent.step
        self._top_score   = self._agent.top_score

        print(
            f"NTupleAgent: loaded '{agent_path}' | "
            f"n={self._n} | "
            f"trained_eps={self._trained_eps:,} | "
            f"top_score={self._top_score:,}"
        )

    def _load_agent(self, agent_path: str, weights_path: str):
        """Load QAgent, trying multiple strategies to handle module deps."""

        # Strategy 1: try loading directly (works if game2048 is on path)
        try:
            with open(agent_path, "rb") as f:
                agent = pickle.load(f)
            with open(weights_path, "rb") as f:
                agent.weights = pickle.load(f)
            agent.np_to_list()
            return agent
        except ModuleNotFoundError:
            pass

        # Strategy 2: walk up directory tree to find game2048 package
        _p = os.path.abspath(agent_path)
        for _ in range(10):
            _p = os.path.dirname(_p)
            if os.path.isdir(os.path.join(_p, 'game2048')):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
                try:
                    with open(agent_path, "rb") as f:
                        agent = pickle.load(f)
                    with open(weights_path, "rb") as f:
                        agent.weights = pickle.load(f)
                    agent.np_to_list()
                    return agent
                except Exception:
                    pass

        # Strategy 3: use a custom unpickler that stubs missing modules
        import types
        import importlib

        # Cache of stub classes so the same (module, name) always returns
        # the identical type object — required for isinstance checks in pickle.
        _stub_cache = {}

        def _make_stub(module, name):
            key = (module, name)
            if key not in _stub_cache:
                _stub_cache[key] = type(name, (object,), {'__module__': module})
            return _stub_cache[key]

        class _StubModule(types.ModuleType):
            """Stub for missing modules during unpickling."""
            def __getattr__(self, name):
                return _make_stub(self.__name__, name)

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                import importlib
                try:
                    mod = importlib.import_module(module)
                    return getattr(mod, name)
                except Exception:
                    return _make_stub(module, name)

        with open(agent_path, "rb") as f:
            agent = _SafeUnpickler(f).load()
        if weights_path:
            with open(weights_path, "rb") as f:
                agent.weights = pickle.load(f)
        # Convert numpy weight arrays to Python lists for faster indexing
        if callable(getattr(agent, 'np_to_list', None)):
            agent.np_to_list()
        elif isinstance(getattr(agent, 'weights', None), (list, tuple)):
            agent.weights = [
                w.tolist() if hasattr(w, 'tolist') else w
                for w in agent.weights
            ]
        # Override the stubbed features function with our embedded implementation
        agent.features = _f_6
        return agent

    def _evaluate(self, log2_board: np.ndarray) -> float:
        """Score a board using the learned weight lookup tables.

        Args:
            log2_board: (4,4) int32 array in log2 representation.

        Returns:
            float: estimated value of this board state.
        """
        return sum(
            self._agent.weights[i][f]
            for i, f in enumerate(self._features(log2_board))
        )

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Pick the action whose resulting board has the highest estimated value.

        Converts the board to log2 format, tries every legal move using
        the embedded move table, scores each result, returns the best action.

        Args:
            state:           (4,4) numpy array with actual tile values.
            available_moves: Non-empty list of valid Action values.
            game_context:    Unused (n-tuple is stateless at inference).

        Returns:
            Action: the chosen move direction.
        """
        log2_board  = _actual_to_log2(state)
        best_action = available_moves[0]
        best_value  = -np.inf

        for action in available_moves:
            direction          = _ACTION_TO_DIR[action]
            new_board, _, changed = _pre_move(log2_board, 0, direction)
            if not changed:
                continue
            value = self._evaluate(new_board)
            if value > best_value:
                best_value  = value
                best_action = action

        return best_action

    def get_params(self) -> Dict:
        return {
            "architecture":  f"n-tuple TD(0), n={self._n}",
            "grid_size":     4,
            "trained_eps":   self._trained_eps,
            "top_score":     self._top_score,
            "agent_path":    self.agent_path,
            "weights_path":  self.weights_path,
        }


# ================================================================
#  CLI -- run eval and write log files
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate NTuple agent and write log files.")
    parser.add_argument("--agent",   required=True,
                        help="Path to agent .pkl")
    parser.add_argument("--weights", required=True,
                        help="Path to weights .pkl")
    parser.add_argument("--games",   type=int, default=100,
                        help="Number of evaluation games")
    parser.add_argument("--config",  default="config.json",
                        help="Path to game config JSON")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory to write log files")
    parser.add_argument("--name",    default="N-Tuple TD Agent",
                        help="Agent display name")
    args = parser.parse_args()

    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    # Load game config (NTuple agent is always 4x4)
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "grid_size": 4,
            "tile_2_probability": 0.9,
            "initial_tiles": 2,
            "random_seed": None,
        }
    config["grid_size"] = 4

    agent = NTupleAgent(
        agent_path=args.agent,
        weights_path=args.weights,
        name=args.name,
    )

    os.makedirs(args.log_dir, exist_ok=True)

    # log_move_detail=True required for per-move reward breakdown metrics
    logger = RunLogger(log_dir=args.log_dir, log_move_detail=True)

    module = InteractionModule(
        config=config,
        agent=agent,
        logger=logger,
        verbose=True,
    )
    module.run(num_games=args.games)
    module.print_results()

    results_path = os.path.join(args.log_dir, "ntuple_agent_eval_results.json")
    module.save_results(results_path)

    # Convert JSONL -> wrapped JSON (matches PPO/DQN log format)
    jsonl_path   = os.path.join(args.log_dir, "ntuple_agent_runs.jsonl")
    wrapped_path = os.path.join(args.log_dir, "ntuple_agent_runs.json")

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
            "agent_type":     "ntuple_agent",
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