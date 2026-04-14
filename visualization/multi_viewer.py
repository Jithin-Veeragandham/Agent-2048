"""
multi_viewer.py
===============

Watch all agents play simultaneously on a single screen.

Each agent runs in its own background **process** (not thread), so the
Python GIL never causes starvation — fast agents (PPO) run at their
true speed independently of slow ones (BeamSearch).

State updates flow from worker processes → main process via a small
Queue, keeping display overhead out of the agent loop.

Usage::

    python visualization/multi_viewer.py

Layout (3×2):
    ┌──────────────┬──────────────┬──────────────┐
    │  Random      │  BeamSearch  │  MCTS        │
    ├──────────────┼──────────────┼──────────────┤
    │  Expectimax  │  PPO         │  N-Tuple TD  │
    └──────────────┴──────────────┴──────────────┘
"""

import os
import sys
import time
import copy
import multiprocessing as mp

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import Game2048, Game2048Visual, Action
from framework.evaluation import REWARD_SEARCH
from agents.base import BaseAgent


# ─── Random agent (module-level so multiprocessing can pickle it) ─

class RandomAgent(BaseAgent):
    agent_type = "random"
    def choose_action(self, state, available_moves, game_context=None):
        return available_moves[np.random.randint(len(available_moves))]
    def get_params(self):
        return {}


# ─── Agent configurations ─────────────────────────────────────────

def _build_agents():
    from agents.beam_search import BeamSearchAgent
    from agents.mcts import MCTSAgent
    from agents.expectimax_snake import ExpectimaxSnakeAgent
    from agents.ppo2048 import PPOAgent
    from agents.ntuple_agent import NTupleAgent

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ppo_ckpt    = os.path.join(_root, 'checkpoints', 'ppo_model.pt')
    ntuple_ckpt = os.path.join(_root, 'checkpoints', 'ntuple_best_agent_portable.pkl')
    ntuple_wts  = None   # portable pkl already contains weights

    return [
        RandomAgent("Random"),
        BeamSearchAgent(beam_width=10, search_depth=15),
        MCTSAgent(num_simulations=300, rollout_depth=15),
        ExpectimaxSnakeAgent(depth=2),
        PPOAgent(model_path=ppo_ckpt if os.path.exists(ppo_ckpt) else None, device='cpu'),
        NTupleAgent(agent_path=ntuple_ckpt, weights_path=ntuple_wts, name="N-Tuple TD"),
    ]


# ─── Layout constants ─────────────────────────────────────────────

COLS            = 3   # 5 agents → 3×2 grid (last slot empty)
ROWS            = 2
CELL_SIZE       = 80
CELL_PAD        = 8
HEADER_H        = 90
PANEL_PAD       = 12
BG_COLOR        = (50, 50, 50)
PANEL_BG        = (187, 173, 160)

TILE_COLORS      = Game2048Visual.TILE_COLORS
DEFAULT_TILE_CLR = Game2048Visual.DEFAULT_TILE_COLOR


# ─── Display-side state (main process only) ───────────────────────

class AgentState:
    """Latest known snapshot of one agent's game (main process only)."""

    def __init__(self, name: str, grid_size: int):
        self.name          = name
        self.grid_size     = grid_size
        self.board         = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.score         = 0
        self.moves         = 0
        self.done          = False
        self.moves_per_sec = 0.0


# ─── Worker process entry point ───────────────────────────────────

def agent_worker(agent, config, queue: mp.Queue, stop_event: mp.Event):
    """Run one complete game and stream state updates into `queue`.

    Runs in a separate process — no GIL sharing with other agents.
    Sends (board, score, moves, mps, done) tuples. Uses put_nowait so
    a slow display loop never blocks the agent loop.
    """
    # Re-import inside worker (needed on Windows spawn)
    import time
    import numpy as np
    from game.engine import Game2048
    from framework.evaluation import REWARD_SEARCH

    game = Game2048(config)
    agent.on_episode_start()

    moves        = 0
    window_moves = 0
    window_start = time.time()
    mps          = 0.0

    while not game.is_game_over() and not stop_event.is_set():
        available = game.get_available_moves()
        if not available:
            break

        board_snap   = game.get_state()
        game_context = {
            'game':        game,
            'score':       game.get_score(),
            'move_number': moves,
            'reward_fn':   REWARD_SEARCH,
        }

        action = agent.choose_action(board_snap, available, game_context)
        valid, reward = game.move(action)

        if valid:
            next_state = game.get_state()
            done       = game.is_game_over()
            agent.on_move_result(board_snap, action, reward, next_state, done)

            window_moves += 1
            elapsed = time.time() - window_start
            if elapsed >= 0.5:
                mps          = window_moves / elapsed
                window_moves = 0
                window_start = time.time()

            moves += 1

            # Non-blocking send — drop update if display is busy
            try:
                queue.put_nowait((game.get_state().copy(), game.get_score(), moves, mps, False))
            except Exception:
                pass

    final_board = game.get_state().copy()
    final_score = game.get_score()
    agent.on_episode_end(final_board, final_score)

    # Final update must get through (blocking)
    queue.put((final_board, final_score, moves, mps, True))


# ─── Panel renderer ───────────────────────────────────────────────

def draw_panel(surface, state: AgentState, rect: pygame.Rect, fonts):
    grid  = state.grid_size
    board = state.board

    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=8)

    x0, y0 = rect.x + 8, rect.y + 6

    surface.blit(fonts['name'].render(state.name, True, (255, 255, 255)), (x0, y0))
    surface.blit(fonts['stat'].render(f"Score: {state.score:,}", True, (255, 255, 255)), (x0, y0 + 26))

    if state.done:
        status_surf = fonts['stat'].render(f"DONE  {state.moves:,} moves", True, (255, 220, 80))
    else:
        mps = state.moves_per_sec
        mps_str = f"{mps:.0f} moves/sec" if mps >= 1 else f"{mps:.2f} moves/sec"
        status_surf = fonts['stat'].render(mps_str, True, (200, 240, 200))
    surface.blit(status_surf, (x0, y0 + 48))

    board_x = rect.x + CELL_PAD
    board_y = rect.y + HEADER_H

    for r in range(grid):
        for c in range(grid):
            val   = int(board[r, c])
            tx    = board_x + c * (CELL_SIZE + CELL_PAD)
            ty    = board_y + r * (CELL_SIZE + CELL_PAD)
            tile  = pygame.Rect(tx, ty, CELL_SIZE, CELL_SIZE)
            color = TILE_COLORS.get(val, DEFAULT_TILE_CLR)
            pygame.draw.rect(surface, color, tile, border_radius=5)
            if val != 0:
                tc  = (119, 110, 101) if val <= 4 else (255, 255, 255)
                num = fonts['tile'].render(str(val), True, tc)
                surface.blit(num, num.get_rect(center=tile.center))

    if state.done:
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        surface.blit(overlay, rect.topleft)
        go_surf = fonts['gameover'].render("GAME OVER", True, (255, 80, 80))
        surface.blit(go_surf, go_surf.get_rect(center=rect.center))


# ─── Main entry point ─────────────────────────────────────────────

def run_multi_visual(agents, config):
    """Launch agents as separate processes and render them in one window."""
    pygame.init()
    grid_size = config.get('grid_size', 4)

    board_px = grid_size * (CELL_SIZE + CELL_PAD) + CELL_PAD
    panel_w  = board_px
    panel_h  = HEADER_H + board_px
    win_w    = COLS * panel_w + (COLS + 1) * PANEL_PAD
    win_h    = ROWS * panel_h + (ROWS + 1) * PANEL_PAD

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("2048 — Agent Race")
    clock  = pygame.time.Clock()

    fonts = {
        'name':     pygame.font.Font(None, 26),
        'stat':     pygame.font.Font(None, 22),
        'tile':     pygame.font.Font(None, max(18, 46 - grid_size * 4)),
        'gameover': pygame.font.Font(None, 40),
    }

    states = [AgentState(agent.name, grid_size) for agent in agents]

    panel_rects = []
    for idx in range(len(agents)):
        row = idx // COLS
        col = idx % COLS
        panel_rects.append(pygame.Rect(
            PANEL_PAD + col * (panel_w + PANEL_PAD),
            PANEL_PAD + row * (panel_h + PANEL_PAD),
            panel_w, panel_h,
        ))

    # One queue + stop event per agent
    queues     = [mp.Queue(maxsize=4) for _ in agents]
    stop_event = mp.Event()

    processes = []
    for agent, queue in zip(agents, queues):
        p = mp.Process(
            target=agent_worker,
            args=(agent, config, queue, stop_event),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print("Running — Q to quit early\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Drain queues — apply latest update per agent
        for state, queue in zip(states, queues):
            if state.done:
                continue
            latest = None
            try:
                while True:
                    latest = queue.get_nowait()
            except Exception:
                pass
            if latest is not None:
                board, score, moves, mps, done = latest
                state.board         = board
                state.score         = score
                state.moves         = moves
                state.moves_per_sec = mps
                state.done          = done

        # Exit once all agents finish
        if all(s.done for s in states):
            screen.fill(BG_COLOR)
            for state, rect in zip(states, panel_rects):
                draw_panel(screen, state, rect, fonts)
            pygame.display.flip()
            pygame.time.delay(1500)
            running = False

        screen.fill(BG_COLOR)
        for state, rect in zip(states, panel_rects):
            draw_panel(screen, state, rect, fonts)
        pygame.display.flip()
        clock.tick(30)

    stop_event.set()
    pygame.quit()

    print(f"\n{'Agent':<30} {'Score':>10}  {'Moves':>8}  {'Status'}")
    print("-" * 60)
    for state in states:
        status = "finished" if state.done else "interrupted"
        print(f"{state.name:<30} {state.score:>10,}  {state.moves:>8,}  {status}")


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    mp.freeze_support()   # needed for Windows packaged builds
    config = {'grid_size': 4, 'tile_2_probability': 0.9, 'initial_tiles': 2}
    agents = _build_agents()
    run_multi_visual(agents, config)
