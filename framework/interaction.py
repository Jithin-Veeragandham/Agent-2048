import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from game import Game2048, Action
from agents.base import BaseAgent
from framework.evaluation import GameEvaluator, RewardFunction, REWARD_SEARCH

# Re-export BaseAgent so old code using `from interaction import BaseAgent` still works
__all__ = ['InteractionModule', 'run_comparison', 'BaseAgent']


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE EPISODE WORKER (top-level for pickling)
# ═══════════════════════════════════════════════════════════════════

<<<<<<< HEAD
def _run_single_episode(config, agent, reward_fn):
=======
def _run_single_episode(config, agent, reward_fn, log_move_detail=False):
>>>>>>> origin/master
    """Play one complete game in a worker process.

    Top-level function so it can be pickled by ProcessPoolExecutor.
    Creates its own Game2048 instance. No logger or evaluator —
    just returns the raw results for the main process to merge.

    Args:
        config: Game configuration dict.
        agent: A BaseAgent instance (must be picklable).
        reward_fn: RewardFunction instance.

    Returns:
        Dict with score, highest_tile, moves, reached_2048,
        total_inference_ms, and per-move inference times.
    """
    game = Game2048(config)
    agent.on_episode_start()

    episode_start = time.time()
    move_number = 0
    inference_times = []
    reward_breakdowns = []

    while not game.is_game_over():
        available = game.get_available_moves()
        if not available:
            break

        state = game.get_state()
        score = game.get_score()

        game_context = {
            'game': game,
            'score': score,
            'move_number': move_number,
            'reward_fn': reward_fn,
        }

        t0 = time.time()
        action = agent.choose_action(state, available, game_context)
        inference_ms = (time.time() - t0) * 1000
        inference_times.append(inference_ms)

        game.move(action)

        next_state = game.get_state()
        done = game.is_game_over()
        agent.on_move_result(state, action, 0, next_state, done)
        reward_breakdowns.append(reward_fn.compute_breakdown(next_state))

        move_number += 1

    final_state = game.get_state()
    final_score = game.get_score()
    highest_tile = int(np.max(final_state))

    agent.on_episode_end(final_state, final_score)

    # Average each breakdown component across all moves
    avg_breakdown = {}
    if reward_breakdowns:
        for key in reward_breakdowns[0]:
            avg_breakdown[key] = float(np.mean([b[key] for b in reward_breakdowns]))

<<<<<<< HEAD
    return {
=======
    # Quartile snapshots for trajectory analysis
    quartile_breakdowns = {}
    if reward_breakdowns:
        n = len(reward_breakdowns)
        indices = {
            'q25':  min(int(n * 0.25), n - 1),
            'q50':  min(int(n * 0.50), n - 1),
            'q75':  min(int(n * 0.75), n - 1),
            'q100': n - 1,
        }
        quartile_breakdowns = {
            label: {k: float(v) for k, v in reward_breakdowns[idx].items()}
            for label, idx in indices.items()
        }

    result = {
>>>>>>> origin/master
        'score': final_score,
        'highest_tile': highest_tile,
        'moves': move_number,
        'reached_2048': highest_tile >= 2048,
        'final_board': final_state,
        'inference_times': inference_times,
        'avg_reward_breakdown': avg_breakdown,
<<<<<<< HEAD
        'game_time_sec': time.time() - episode_start,
    }
=======
        'quartile_reward_breakdowns': quartile_breakdowns,
        'game_time_sec': time.time() - episode_start,
    }
    if log_move_detail:
        result['move_reward_breakdowns'] = reward_breakdowns
    return result
>>>>>>> origin/master


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
        logger: Optional RunLogger instance from framework/logger.py. If provided,
            board states and reward breakdowns are logged every move.
        verbose: If True, print per-episode stats during run.
        num_workers: Number of parallel workers for episode execution.
            1 = sequential (default). >1 = parallel episodes via
            ProcessPoolExecutor. Parallel mode disables per-move
            logging and print_board since those require sequential
            access. The agent must be picklable for parallel mode.

    Example::

        module = InteractionModule(
            config={"grid_size": 6},
            agent=my_agent,
            reward_fn=REWARD_SEARCH,
        )
        module.run(num_games=100)
        module.print_results()
        module.save_results("mcts_6x6_results.json")

    Example with parallel episodes::

        module = InteractionModule(
            config={"grid_size": 4},
            agent=my_agent,
            num_workers=8,       # 8 games at once
            verbose=True,
        )
        module.run(num_games=100)
        module.print_results()

    Example with logging (sequential only)::

        from framework.logger import RunLogger

        logger = RunLogger()
        module = InteractionModule(
            config={"grid_size": 4},
            agent=my_agent,
            reward_fn=REWARD_SEARCH,
            logger=logger,
        )
        module.run(num_games=50)
    """

    def __init__(
        self,
        config: Dict,
        agent: BaseAgent,
        reward_fn: Optional[RewardFunction] = None,
        logger=None,
        verbose: bool = False,
        print_board: bool = False,
        num_workers: int = 1,
    ):
        self.config = config
        self.agent = agent
        self.reward_fn = reward_fn or REWARD_SEARCH
        self.evaluator = GameEvaluator()
        self.logger = logger
        self.verbose = verbose
        self.print_board = print_board
        self.num_workers = max(1, num_workers)

    # ─── Single Episode (sequential, with full logging) ───────────

    def run_episode(self) -> Dict:
        """Play one complete game, tracking everything automatically.

        Flow per move:
            1. Get state + available moves from Game Module
            2. Pass to agent via choose_action()
            3. Record inference time
            4. Execute action on game, get reward
            5. Notify agent of transition (for RL agents)
            6. Log move to evaluator (and logger if present)

        Returns:
            Dict with episode summary: score, highest_tile, moves, etc.
        """
        game = Game2048(self.config)
        self.evaluator.start_episode()
        self.agent.on_episode_start()

        if self.logger:
            self.logger.on_episode_start()

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

            # ── Log to logger (if present) ──
            if self.logger:
<<<<<<< HEAD
                breakdown = self.reward_fn.compute_breakdown(state)
=======
                breakdown = self.reward_fn.compute_breakdown(next_state)
>>>>>>> origin/master
                self.logger.log_move(
                    step=move_number,
                    state=state,
                    action=action,
                    reward=reward,
                    score=game.get_score(),
                    inference_ms=inference_ms,
                    reward_breakdown=breakdown,
                )

            # ── Print per-move board state ──
            if self.print_board:
                max_tile = int(np.max(next_state))
                print(
                    f"  Step {move_number:>4}  |  {action.name:<5}  |  "
                    f"Reward: {reward:>5}  |  Score: {game.get_score():>7}  |  "
                    f"Max tile: {max_tile:>5}  |  {inference_ms:>7.1f} ms"
                )
                print(next_state)
                print()

            move_number += 1

        # ── Episode complete ──
        final_state = game.get_state()
        final_score = game.get_score()

        self.agent.on_episode_end(final_state, final_score)
        ep_stats = self.evaluator.end_episode(game)

        # ── Finalize logger episode ──
        if self.logger:
            final_breakdown = self.reward_fn.compute_breakdown(final_state)
            self.logger.end_episode(
                final_score=final_score,
                highest_tile=ep_stats.highest_tile,
                move_count=ep_stats.move_count,
                reached_2048=ep_stats.reached_2048,
                final_board=final_state,
                final_reward_breakdown=final_breakdown,
            )

        return {
            'score': ep_stats.score,
            'highest_tile': ep_stats.highest_tile,
            'moves': ep_stats.move_count,
            'reached_2048': ep_stats.reached_2048,
            'final_board': final_state,
        }

    # ─── Multi-Episode Run ────────────────────────────────────────

    def run(self, num_games: int = 100) -> List[Dict]:
        """Run multiple games and collect evaluation data.

        Uses sequential execution when num_workers=1, or parallel
        execution via ProcessPoolExecutor when num_workers>1.

        Parallel mode skips per-move logging and print_board since
        those require sequential access. Episode-level stats are
        still collected and merged into the evaluator.

        Args:
            num_games: Number of complete games to play.

        Returns:
            List of per-episode summary dicts.
        """
        if self.logger:
            self.logger.on_run_start(
                num_games=num_games,
                agent_name=self.agent.name,
                agent_type=self.agent.agent_type,
                agent_params=self.agent.get_params(),
                config=self.config,
                reward_weights=self.reward_fn.weights,
            )

        parallel = self.num_workers > 1
        mode_label = f", {self.num_workers} workers" if parallel else ""
        print(f"\nRunning {self.agent.name} for {num_games} games "
              f"(grid: {self.config.get('grid_size', 4)}×"
              f"{self.config.get('grid_size', 4)}{mode_label})...")
<<<<<<< HEAD
        print(f"{'─' * 50}")
=======
        print("-" * 50)
>>>>>>> origin/master

        if parallel:
            results = self._run_parallel(num_games)
        else:
            results = self._run_sequential(num_games)

<<<<<<< HEAD
        print(f"{'─' * 50}")
=======
        print("-" * 50)
>>>>>>> origin/master
        print(f"Done.\n")

        if self.logger:
            self.logger.save()

        return results

    def _run_sequential(self, num_games: int) -> List[Dict]:
        """Run games one at a time with full logging support."""
        results = []
        for i in range(num_games):
            ep_result = self.run_episode()
            results.append(ep_result)

            if self.verbose:
                print(f"  Game {i + 1:>4}/{num_games}  |  "
                      f"Score: {ep_result['score']:>8}  |  "
                      f"Max Tile: {ep_result['highest_tile']:>5}  |  "
                      f"Moves: {ep_result['moves']:>4}")
                if self.print_board:
                    print(ep_result.get('final_board', ''))
                    print()
            elif (i + 1) % max(1, num_games // 10) == 0:
                pct = (i + 1) / num_games * 100
                print(f"  Progress: {pct:.0f}% ({i + 1}/{num_games})")

        return results

    def _run_parallel(self, num_games: int) -> List[Dict]:
        """Run games in parallel via ProcessPoolExecutor.

        Each worker gets its own Game2048 and runs the agent
        independently. Results are merged back into the evaluator
        on the main process. Per-move logging is skipped.
        """
        results = [None] * num_games
        completed = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}
<<<<<<< HEAD
=======
            log_move_detail = self.logger.log_move_detail if self.logger else False
>>>>>>> origin/master
            for i in range(num_games):
                f = pool.submit(
                    _run_single_episode,
                    self.config,
                    self.agent,
                    self.reward_fn,
<<<<<<< HEAD
=======
                    log_move_detail,
>>>>>>> origin/master
                )
                futures[f] = i

            for f in as_completed(futures):
                idx = futures[f]
                ep_result = f.result()
                results[idx] = ep_result
                completed += 1

                # Merge into evaluator: replay the inference times
                self.evaluator.start_episode()
                for ms in ep_result['inference_times']:
                    self.evaluator.log_move(ms)
                # Build a minimal game for end_episode
                final_game = Game2048.from_state(
                    ep_result['final_board'],
                    score=ep_result['score'],
                )
                self.evaluator.end_episode(final_game)

                if self.logger:
                    final_board_arr = np.array(ep_result['final_board'])
                    final_breakdown = self.reward_fn.compute_breakdown(final_board_arr)
                    self.logger.on_episode_start()
                    self.logger._game_start = time.time() - ep_result['game_time_sec']
                    self.logger._current_inference_times = ep_result['inference_times']
                    self.logger._current_moves = [{'reward_breakdown': ep_result['avg_reward_breakdown']}]
                    self.logger.end_episode(
                        final_score=ep_result['score'],
                        highest_tile=ep_result['highest_tile'],
                        move_count=ep_result['moves'],
                        reached_2048=ep_result['reached_2048'],
                        final_board=final_board_arr,
                        final_reward_breakdown=final_breakdown,
<<<<<<< HEAD
=======
                        quartile_reward_breakdowns=ep_result.get('quartile_reward_breakdowns', {}),
                        move_reward_breakdowns=ep_result.get('move_reward_breakdowns'),
>>>>>>> origin/master
                    )

                if self.verbose:
                    print(f"  Game {completed:>4}/{num_games}  |  "
                          f"Score: {ep_result['score']:>8}  |  "
                          f"Max Tile: {ep_result['highest_tile']:>5}  |  "
                          f"Moves: {ep_result['moves']:>4}")
                elif completed % max(1, num_games // 10) == 0:
                    pct = completed / num_games * 100
                    print(f"  Progress: {pct:.0f}% ({completed}/{num_games})")

        # Strip worker-only fields before returning
        for r in results:
            r.pop('inference_times', None)

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
    logger=None,
    verbose: bool = False,
    num_workers: int = 1,
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
        logger: Optional RunLogger. If provided, each agent's run
            is logged to the appropriate agent-type jsonl file.
        verbose: Print per-game stats.
        num_workers: Parallel workers for episode execution.

    Returns:
        List of summary dicts, one per agent.

    Example::

        from framework.interaction import run_comparison

        results = run_comparison(
            config={"grid_size": 6},
            agents=[mcts_agent, dqn_agent, ppo_agent, alphazero_agent, beam_agent],
            num_games=100,
            num_workers=8,
        )
    """
    from framework.evaluation import compare_agents

    all_results = []

    for agent in agents:
        module = InteractionModule(
            config, agent, reward_fn, logger, verbose,
            num_workers=num_workers,
        )
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
<<<<<<< HEAD
    module.print_results()
=======
    module.print_results()
>>>>>>> origin/master
