# coding: utf-8
"""
agent_viewer.py
===============

Visualization module: watch any 2048 agent play in real time.

Reuses ``Game2048Visual`` from ``game/engine.py`` for rendering —
the only new logic here is the agent-driven game loop and CLI.

Usage::

    python visualization/agent_viewer.py --agent beam_search --width 10 --depth 15
    python visualization/agent_viewer.py --agent mcts --simulations 100 --delay 500
    python visualization/agent_viewer.py --agent expectimax --search-depth 3
    python visualization/agent_viewer.py --agent ntuple
    python visualization/agent_viewer.py --agent random

Controls:
    SPACE  — pause / resume
    R      — reset to a new game
    Q      — quit
"""

import os
import sys
import argparse
import importlib

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

# Allow running directly from repo root or from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import Game2048, Game2048Visual, load_config
from framework.evaluation import REWARD_SEARCH


# ─── Agent registry ───────────────────────────────────────────────

AGENT_REGISTRY = {
    'beam_search':     ('agents.beam_search',      'BeamSearchAgent'),
    'mcts':            ('agents.mcts',              'MCTSAgent'),
    'expectimax':      ('agents.expectimax',        'ExpectimaxAgent'),
    'expectimax_snake':('agents.expectimax_snake',  'ExpectimaxSnakeAgent'),
    'ntuple':          ('agents.ntuple_agent',      'NTupleAgent'),
    'random':          None,   # built-in, no import needed
}


# ─── Core visual loop ─────────────────────────────────────────────

def run_agent_visual(agent, config=None, step_delay_ms=300):
    """Drive Game2048Visual with any BaseAgent.

    Runs the pygame window until the user presses Q or closes it.
    The agent plays automatically; the user can pause, reset, or quit.

    Args:
        agent:          Any BaseAgent instance.
        config:         Game config dict. Defaults to standard 4x4 settings.
        step_delay_ms:  Milliseconds to wait between agent moves.
    """
    if config is None:
        config = {'grid_size': 4, 'tile_2_probability': 0.9, 'initial_tiles': 2}

    game = Game2048(config)
    visual = Game2048Visual(game, config)
    reward_fn = REWARD_SEARCH
    clock = pygame.time.Clock()

    move_number = 0
    paused = False
    running = True

    agent.on_episode_start()
    visual.draw()

    while running:
        # ── Event handling ──────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    agent.on_episode_start()
                    move_number = 0
                    paused = False
                    visual.draw()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"[{status}]  Score: {game.get_score()}")

        if not running:
            break

        # ── Agent step ──────────────────────────────────────────
        if not paused and not game.is_game_over():
            available = game.get_available_moves()
            if available:
                state = game.get_state()
                game_context = {
                    'game': game,
                    'score': game.get_score(),
                    'move_number': move_number,
                    'reward_fn': reward_fn,
                }
                action = agent.choose_action(state, available, game_context)
                valid, reward = game.move(action)

                if valid:
                    next_state = game.get_state()
                    done = game.is_game_over()
                    agent.on_move_result(state, action, reward, next_state, done)
                    move_number += 1

                visual.draw()
                pygame.time.delay(step_delay_ms)

        elif game.is_game_over():
            # Keep rendering the game-over overlay; wait for R or Q
            visual.draw()
            pygame.time.delay(100)

        clock.tick(60)

    final_state = game.get_state()
    agent.on_episode_end(final_state, game.get_score())
    visual.close()
    print(f"\nFinal score: {game.get_score()}  |  Moves: {move_number}")


# ─── Agent factory ────────────────────────────────────────────────

def _build_agent(args):
    """Instantiate the agent requested by CLI args."""
    if args.agent == 'random':
        from framework.interaction import RandomAgent
        return RandomAgent()

    module_path, class_name = AGENT_REGISTRY[args.agent]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    if args.agent == 'beam_search':
        return cls(beam_width=args.width, search_depth=args.depth)
    elif args.agent == 'mcts':
        return cls(num_simulations=args.simulations)
    elif args.agent in ('expectimax', 'expectimax_snake'):
        return cls(depth=args.search_depth)
    elif args.agent == 'ntuple':
        return cls(
            agent_path=args.ntuple_agent,
            weights_path=args.ntuple_weights,
        )
    else:
        return cls()


# ─── CLI entry point ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Watch a 2048 agent play in real time.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--agent', choices=list(AGENT_REGISTRY.keys()), default='random',
        help='Which agent to run.',
    )
    parser.add_argument('--delay',       type=int, default=300,
                        help='Milliseconds between moves.')
    parser.add_argument('--config',      type=str, default=None,
                        help='Path to a JSON config file.')

    # BeamSearch
    parser.add_argument('--width',       type=int, default=10,
                        help='[beam_search] Beam width.')
    parser.add_argument('--depth',       type=int, default=15,
                        help='[beam_search] Search depth.')

    # MCTS
    parser.add_argument('--simulations', type=int, default=200,
                        help='[mcts] Number of simulations per move.')

    # Expectimax
    parser.add_argument('--search-depth', dest='search_depth', type=int, default=2,
                        help='[expectimax / expectimax_snake] Search depth.')

    # NTuple
    parser.add_argument('--ntuple-agent',   dest='ntuple_agent',
                        default='checkpoints/ntuple_best_agent.pkl',
                        help='[ntuple] Path to agent .pkl')
    parser.add_argument('--ntuple-weights', dest='ntuple_weights',
                        default='checkpoints/ntuple_best_agent_weights.pkl',
                        help='[ntuple] Path to weights .pkl')

    args = parser.parse_args()

    config = load_config(args.config or 'config.json')
    agent  = _build_agent(args)

    print(f"Agent: {agent.name}  |  Delay: {args.delay} ms  |  "
          f"Grid: {config.get('grid_size', 4)}x{config.get('grid_size', 4)}")
    print("Controls: SPACE=pause  R=reset  Q=quit\n")

    run_agent_visual(agent, config, step_delay_ms=args.delay)