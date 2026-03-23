from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import numpy as np

from game import Action


class BaseAgent(ABC):
    """Abstract base class for all 2048 agents.

    Every agent (MCTS, DQN, PPO, AlphaZero, Beam Search) must
    subclass this and implement ``choose_action()``.

    The Interaction Module calls ``choose_action()`` at every step,
    passing the current board state, list of valid moves, and an
    optional game context for agents that need deeper access
    (e.g., MCTS needs to clone the game for simulations).

    Attributes:
        name: Human-readable agent name (used in eval reports).
        agent_type: Slug used for log filename (override in subclasses).

    Example::

        class RandomAgent(BaseAgent):
            def choose_action(self, state, available_moves, game_context):
                return available_moves[np.random.randint(len(available_moves))]
    """

    # Override in subclasses: used as the log filename slug
    # e.g. "beam_search" -> logs/beam_search_runs.jsonl
    agent_type: str = "agent"

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Select an action given the current game state.

        Args:
            state: Current board as a 2D numpy array. This is a copy —
                modifying it won't affect the real game.
            available_moves: List of valid Action values that would
                change the board. Guaranteed non-empty.
            game_context: Optional dict with extra info:
                - 'game': the Game2048 instance (for cloning in search agents)
                - 'score': current cumulative score
                - 'move_number': how many moves have been played
                - 'reward_fn': the RewardFunction instance

        Returns:
            Action: The chosen move direction.
        """
        ...

    def on_episode_start(self):
        """Called at the start of each game. Override for per-game setup."""
        pass

    def on_episode_end(self, final_state: np.ndarray, score: int):
        """Called at the end of each game. Override for learning/logging.

        Args:
            final_state: The final board state.
            score: The final merge score.
        """
        pass

    def on_move_result(self, state: np.ndarray, action: Action,
                       reward: int, next_state: np.ndarray, done: bool):
        """Called after each move with the transition tuple.

        This is the hook for RL agents to collect experience:
            (s, a, r, s', done)

        Search agents can ignore this.

        Args:
            state: Board before the move.
            action: Action that was taken.
            reward: Reward received from the move.
            next_state: Board after the move (and tile spawn).
            done: Whether the game is over.
        """
        pass

    def get_params(self) -> Dict:
        """Return agent-specific parameters for logging.

        Override in subclasses to report architecture params like
        beam_width, search_depth, num_simulations, etc.

        Returns:
            Dict of param_name -> value.
        """
        return {}
