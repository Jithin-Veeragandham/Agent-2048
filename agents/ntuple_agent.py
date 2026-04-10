"""
agents/ntuple_agent.py
================

Wraps the tabular QAgent (n-tuple TD learning) as a BaseAgent so it
plugs directly into the Agent-2048 InteractionModule and RunLogger.

Architecture
------------
N-tuple TD(0) learning with a linear value function over hand-crafted
board features (rows, columns, 3x3 squares). The agent uses a greedy
policy ΓÇö no exploration at inference time. Action selection works by
trying all 4 legal moves, scoring each resulting board state with the
learned weight table, and picking the highest-scoring one.

Important: the QAgent's internal Game class stores tiles as log2 values
(0=empty, 1=tile-2, 2=tile-4, ...), while Agent-2048's Game2048 stores
actual tile values (0, 2, 4, 8, ...). This wrapper converts between the
two representations at every step.

Usage
-----
    from agents.ntuple_agent import NTupleAgent
    from framework.interaction import InteractionModule
    from framework.logger import RunLogger

    agent  = NTupleAgent(agent_path="game2048/local_storage/a/my_agent.pkl",
                        weights_path="game2048/local_storage/weights/my_agent.pkl")
    logger = RunLogger()
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
    python agents/ntuple_agent.py --agent game2048/local_storage/a/my_agent.pkl \\
                            --weights game2048/local_storage/weights/my_agent.pkl \\
                            --games 100
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any

# Ensure repo root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from game import Action
from agents.base import BaseAgent

# Action mapping:
# QAgent direction: 0=left, 1=up, 2=right, 3=down
# Agent-2048 Action enum: UP=0, DOWN=1, LEFT=2, RIGHT=3
_DIR_TO_ACTION = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]
_ACTION_TO_DIR = {
    Action.LEFT:  0,
    Action.UP:    1,
    Action.RIGHT: 2,
    Action.DOWN:  3,
}


def _actual_to_log2(board: np.ndarray) -> np.ndarray:
    """Convert Agent-2048 board (actual values) ΓåÆ QAgent board (log2 values).

    Agent-2048: 0=empty, 2=tile-2, 4=tile-4, ...
    QAgent:     0=empty, 1=tile-2, 2=tile-4, ...
    """
    result = np.zeros_like(board, dtype=np.int32)
    nonzero = board > 0
    result[nonzero] = np.log2(board[nonzero]).astype(np.int32)
    return result


class NTupleAgent(BaseAgent):
    """Tabular QAgent (n-tuple TD learning) wrapped as a BaseAgent.

    Loads a trained checkpoint from local_storage and runs greedy
    inference using the learned value function.

    Args:
        agent_path:   Path to agent .pkl (structure without weights).
        weights_path: Path to weights .pkl (separate file).
        name:         Display name for logs.
    """

    agent_type: str = "ntuple_agent"

    def __init__(
        self,
        agent_path: str,
        weights_path: str,
        name: str = "N-Tuple TD Agent",
    ):
        super().__init__(name)
        self.agent_path   = agent_path
        self.weights_path = weights_path

        # The pickle was saved from the 2048/ directory which contains
        # the game2048 package. Walk up from agent_path to find the directory
        # that contains the game2048 package and add it to sys.path.
        # agent_path: .../2048/game2048/local_storage/a/best_agent.pkl
        # We need:    .../2048/
        _p = os.path.abspath(agent_path)
        for _ in range(6):  # walk up at most 6 levels
            _p = os.path.dirname(_p)
            if os.path.isdir(os.path.join(_p, 'game2048')):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
                break

        # Load agent and weights
        with open(agent_path, "rb") as f:
            self._agent = pickle.load(f)
        with open(weights_path, "rb") as f:
            self._agent.weights = pickle.load(f)
        self._agent.np_to_list()

        self._n            = self._agent.n
        self._num_feat     = self._agent.num_feat
        self._features     = self._agent.features
        self._trained_eps  = self._agent.step
        self._top_score    = self._agent.top_score

        print(
            f"NTupleAgent: loaded '{agent_path}' | "
            f"n={self._n} | "
            f"trained_eps={self._trained_eps:,} | "
            f"top_score={self._top_score:,}"
        )

    def _evaluate(self, log2_board: np.ndarray) -> float:
        """Score a board state using the learned weight table.

        Args:
            log2_board: (4,4) int32 array in log2 representation.

        Returns:
            float: estimated value of this board state.
        """
        return sum(
            self._agent.weights[i][f]
            for i, f in enumerate(self._features(log2_board))
        )

    def _pre_move(self, log2_board: np.ndarray, direction: int):
        """Apply a move to a log2 board without modifying it.

        Uses the QAgent's own lookup-table based move logic which
        operates on log2 values. Returns (new_board, changed).
        """
        from game2048.game_logic import Game
        _g = Game.__new__(Game)   # create instance without __init__
        new_board, _, changed = _g.pre_move(log2_board, 0, direction)
        return new_board, changed

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Pick the action whose resulting board has the highest estimated value.

        Converts the Agent-2048 board (actual tile values) to log2
        representation, tries every legal move using the QAgent's move
        table, scores each result, and returns the best action.

        Args:
            state:           (4,4) numpy array with actual tile values.
            available_moves: Non-empty list of valid Action values.
            game_context:    Unused.

        Returns:
            Action: the chosen move direction.
        """
        log2_board = _actual_to_log2(state)

        best_action = available_moves[0]
        best_value  = -np.inf

        for action in available_moves:
            direction        = _ACTION_TO_DIR[action]
            new_board, changed = self._pre_move(log2_board, direction)
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


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  CLI ΓÇö run eval and write log files
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate QAgent and write log files in PPO-compatible format.")
    parser.add_argument("--agent",   required=True,
                        help="Path to agent .pkl (e.g. game2048/local_storage/a/my_agent.pkl)")
    parser.add_argument("--weights", required=True,
                        help="Path to weights .pkl (e.g. game2048/local_storage/weights/my_agent.pkl)")
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

    # Load game config
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
    config["grid_size"] = 4   # QAgent is 4├ù4 only

    agent = NTupleAgent(
        agent_path=args.agent,
        weights_path=args.weights,
        name=args.name,
    )

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

    results_path = os.path.join(args.log_dir, "ntuple_agent_eval_results.json")
    module.save_results(results_path)

    # ΓöÇΓöÇ Convert JSONL ΓåÆ wrapped JSON (matches PPO/DQN log format) ΓöÇ
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
