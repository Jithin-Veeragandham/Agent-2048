import time
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

from game import Game2048, Action
from evaluation import GameEvaluator, RewardFunction, REWARD_SEARCH


# ═══════════════════════════════════════════════════════════════════
#  BASE AGENT INTERFACE — all agents must implement this
# ═══════════════════════════════════════════════════════════════════

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

    Example::

        class RandomAgent(BaseAgent):
            def choose_action(self, state, available_moves, game_context):
                return available_moves[np.random.randint(len(available_moves))]
    """

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


# ═══════════════════════════════════════════════════════════════════
#  INTERACTION MODULE — the central bridge
# ═══════════════════════════════════════════════════════════════════

class InteractionModule:
    """Central bridge between Game, Agent, and Evaluation modules.

    Handles the full game loop: passes state to the agent, relays
    actions to the game, triggers evaluation after every move, and
    collects all metrics automatically.

    Agents don't need to know about evaluation — they just implement
    ``choose_action()`` and the Interaction Module does the rest.

    Args:
        config: Game configuration dict (grid_size, tile_2_probability, etc.).
        agent: An instance of BaseAgent.
        reward_fn: RewardFunction instance for board evaluation.
            Defaults to REWARD_SEARCH.
        verbose: If True, print per-episode stats during run.

    Example::

        module = InteractionModule(
            config={"grid_size": 6},
            agent=my_agent,
            reward_fn=REWARD_SEARCH,
        )
        module.run(num_games=100)
        module.print_results()
        module.save_results("mcts_6x6_results.json")
    """

    def __init__(
        self,
        config: Dict,
        agent: BaseAgent,
        reward_fn: Optional[RewardFunction] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.agent = agent
        self.reward_fn = reward_fn or REWARD_SEARCH
        self.evaluator = GameEvaluator()
        self.verbose = verbose

    # ─── Single Episode ───────────────────────────────────────────

    def run_episode(self) -> Dict:
        """Play one complete game, tracking everything automatically.

        Flow per move:
            1. Get state + available moves from Game Module
            2. Pass to agent via choose_action()
            3. Record inference time
            4. Execute action on game, get reward
            5. Notify agent of transition (for RL agents)
            6. Log move to evaluator

        Returns:
            Dict with episode summary: score, highest_tile, moves, etc.
        """
        game = Game2048(self.config)
        self.evaluator.start_episode()
        self.agent.on_episode_start()

        move_number = 0

        while not game.is_game_over():
            available = game.get_available_moves()
            if not available:
                break

            state = game.get_state()
            score = game.get_score()

            # Build context for agents that need deeper access
            game_context = {
                'game': game,             # for cloning (MCTS, Beam Search)
                'score': score,
                'move_number': move_number,
                'reward_fn': self.reward_fn,
            }

            # ── Agent decides ──
            t0 = time.time()
            action = self.agent.choose_action(state, available, game_context)
            inference_ms = (time.time() - t0) * 1000

            # ── Execute on game ──
            valid, reward = game.move(action)

            # ── Notify agent (for RL experience collection) ──
            next_state = game.get_state()
            done = game.is_game_over()
            self.agent.on_move_result(state, action, reward, next_state, done)

            # ── Log to evaluator ──
            self.evaluator.log_move(inference_ms)
            move_number += 1

        # ── Episode complete ──
        final_state = game.get_state()
        final_score = game.get_score()

        self.agent.on_episode_end(final_state, final_score)
        ep_stats = self.evaluator.end_episode(game)

        return {
            'score': ep_stats.score,
            'highest_tile': ep_stats.highest_tile,
            'moves': ep_stats.move_count,
            'reached_2048': ep_stats.reached_2048,
        }

    # ─── Multi-Episode Run ────────────────────────────────────────

    def run(self, num_games: int = 100) -> List[Dict]:
        """Run multiple games and collect evaluation data.

        Args:
            num_games: Number of complete games to play.

        Returns:
            List of per-episode summary dicts.
        """
        results = []
        print(f"\nRunning {self.agent.name} for {num_games} games "
              f"(grid: {self.config.get('grid_size', 4)}×"
              f"{self.config.get('grid_size', 4)})...")
        print(f"{'─' * 50}")

        for i in range(num_games):
            ep_result = self.run_episode()
            results.append(ep_result)

            if self.verbose:
                print(f"  Game {i + 1:>4}/{num_games}  |  "
                      f"Score: {ep_result['score']:>8}  |  "
                      f"Max Tile: {ep_result['highest_tile']:>5}  |  "
                      f"Moves: {ep_result['moves']:>4}")
            elif (i + 1) % max(1, num_games // 10) == 0:
                pct = (i + 1) / num_games * 100
                print(f"  Progress: {pct:.0f}% ({i + 1}/{num_games})")

        print(f"{'─' * 50}")
        print(f"Done.\n")
        return results

    # ─── Training Support (for RL agents) ─────────────────────────

    def set_training_stats(self, training_time_sec: float, training_episodes: int):
        """Record training cost for RL agents.

        Call this BEFORE run() so the efficiency metrics include
        training cost in the denominator.

        Args:
            training_time_sec: Wall-clock seconds spent training.
            training_episodes: Number of episodes during training.
        """
        self.evaluator.set_training_stats(training_time_sec, training_episodes)

    # ─── Results ──────────────────────────────────────────────────

    def get_results(self) -> Dict:
        """Get full evaluation summary dict."""
        return self.evaluator.get_summary(self.agent.name)

    def print_results(self):
        """Pretty-print evaluation summary."""
        self.evaluator.print_summary(self.agent.name)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        self.evaluator.save_results(filepath)

    def reset(self):
        """Clear evaluation data for a new run."""
        self.evaluator.reset()


# ═══════════════════════════════════════════════════════════════════
#  MULTI-AGENT RUNNER — compare all agents under identical settings
# ═══════════════════════════════════════════════════════════════════

def run_comparison(
    config: Dict,
    agents: List[BaseAgent],
    num_games: int = 100,
    reward_fn: Optional[RewardFunction] = None,
    verbose: bool = False,
) -> List[Dict]:
    """Run multiple agents under identical settings and compare.

    This is the top-level function for your final evaluation.
    Each agent plays the same number of games with the same config,
    and results are printed in a side-by-side comparison table.

    Args:
        config: Game configuration dict.
        agents: List of BaseAgent instances to evaluate.
        num_games: Games per agent.
        reward_fn: Shared RewardFunction (or None for default).
        verbose: Print per-game stats.

    Returns:
        List of summary dicts, one per agent.

    Example::

        from interaction import run_comparison

        results = run_comparison(
            config={"grid_size": 6},
            agents=[mcts_agent, dqn_agent, ppo_agent, alphazero_agent, beam_agent],
            num_games=100,
        )
    """
    from evaluation import compare_agents

    all_results = []

    for agent in agents:
        module = InteractionModule(config, agent, reward_fn, verbose)
        module.run(num_games)
        summary = module.get_results()
        all_results.append(summary)

    # Side-by-side comparison
    compare_agents(all_results)

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  BUILT-IN AGENTS — Random baseline + Human wrapper
# ═══════════════════════════════════════════════════════════════════

class RandomAgent(BaseAgent):
    """Baseline agent that picks uniformly random valid moves.

    Use this to establish a performance floor for comparison.
    """

    def __init__(self):
        super().__init__("Random")

    def choose_action(self, state, available_moves, game_context=None):
        return available_moves[np.random.randint(len(available_moves))]


# ═══════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print(f"Loaded config.json (grid: {config.get('grid_size', 4)}×"
              f"{config.get('grid_size', 4)})")
    except FileNotFoundError:
        config = {"grid_size": 4}
        print("No config.json found, using defaults (4×4)")

    # Run random agent baseline
    agent = RandomAgent()
    module = InteractionModule(config, agent, verbose=True)
    module.run(num_games=20)
    module.print_results()