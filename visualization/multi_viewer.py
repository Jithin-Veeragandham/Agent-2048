"""
multi_viewer.py
===============

Watch all agents play simultaneously on a single screen.

Each agent runs in its own background thread and plays one complete game.
The display updates at ~30 fps until every agent finishes or Q is pressed.
Final scores and move counts are printed to the terminal on exit.

Usage::

    python visualization/multi_viewer.py

Layout (2×2 default):
    ┌─────────────────────┬─────────────────────┐
    │  BeamSearch(w=10)   │      MCTS(n=200)    │
    │  Score: 18,432      │  Score: 12,048      │
    │  47 moves/sec       │  3 moves/sec        │
    │  [board]            │  [board]            │
    ├─────────────────────┼─────────────────────┤
    │  Expectimax(d=3)    │      Random         │
    │  Score: 1,024       │  Score: 980         │
    │  12 moves/sec       │  890 moves/sec      │
    │  [board]            │  [board]            │
    └─────────────────────┴─────────────────────┘
"""

import os
import sys
import time
import threading
import copy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import Game2048, Game2048Visual, Action
from framework.evaluation import REWARD_SEARCH
from framework.interaction import RandomAgent


# ─── Agent configurations ─────────────────────────────────────────

def _build_agents():
    from agents.beam_search import BeamSearchAgent
    from agents.mcts import MCTSAgent
    from agents.expectimax_snake import ExpectimaxSnakeAgent

    return [
        BeamSearchAgent(beam_width=10, search_depth=15),
        MCTSAgent(num_simulations=300, rollout_depth=15),
        ExpectimaxSnakeAgent(depth=2),
        RandomAgent(),
    ]


# ─── Layout constants ─────────────────────────────────────────────

COLS        = 2          # panels across
ROWS        = 2          # panels down
CELL_SIZE   = 80         # px per tile
CELL_PAD    = 8          # px between tiles
HEADER_H    = 90         # px above the board for text
PANEL_PAD   = 12         # px between panels
BG_COLOR    = (50, 50, 50)
PANEL_BG    = (187, 173, 160)
DONE_TINT   = (30, 30, 30)   # dark overlay when game is over

TILE_COLORS     = Game2048Visual.TILE_COLORS
DEFAULT_TILE_CLR= Game2048Visual.DEFAULT_TILE_COLOR


# ─── Shared state ─────────────────────────────────────────────────

class AgentState:
    """Thread-safe snapshot of one agent's game."""

    def __init__(self, name: str, grid_size: int):
        self.name          = name
        self.grid_size     = grid_size
        self.board         = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.score         = 0
        self.moves         = 0
        self.done          = False
        self.moves_per_sec = 0.0   # rolling moves/sec
        self.lock          = threading.Lock()


# ─── Agent thread ─────────────────────────────────────────────────

def agent_loop(agent, config, state: AgentState, stop_event: threading.Event):
    """Play one complete game, writing results into `state` after every move."""
    game = Game2048(config)
    agent.on_episode_start()

    window_moves = 0
    window_start = time.time()

    while not game.is_game_over() and not stop_event.is_set():
        available = game.get_available_moves()
        if not available:
            break

        board_snap = game.get_state()
        game_context = {
            'game':        game,
            'score':       game.get_score(),
            'move_number': state.moves,
            'reward_fn':   REWARD_SEARCH,
        }

        action = agent.choose_action(board_snap, available, game_context)
        valid, reward = game.move(action)

        if valid:
            next_state = game.get_state()
            done = game.is_game_over()
            agent.on_move_result(board_snap, action, reward, next_state, done)

            window_moves += 1
            elapsed = time.time() - window_start
            if elapsed >= 0.5:                        # update speed every 0.5s
                mps = window_moves / elapsed
                window_moves = 0
                window_start = time.time()
            else:
                mps = state.moves_per_sec             # keep last value

            with state.lock:
                state.board         = game.get_state().copy()
                state.score         = game.get_score()
                state.moves        += 1
                state.moves_per_sec = mps

    final_state = game.get_state()
    agent.on_episode_end(final_state, game.get_score())

    with state.lock:
        state.board = game.get_state().copy()
        state.score = game.get_score()
        state.done  = True


# ─── Panel renderer ───────────────────────────────────────────────

def draw_panel(surface, state_snap, rect: pygame.Rect, fonts):
    """Render one agent's panel into `rect` on `surface`."""
    grid   = state_snap.grid_size
    board  = state_snap.board
    score  = state_snap.score
    moves  = state_snap.moves
    mps    = state_snap.moves_per_sec
    done   = state_snap.done

    # Panel background
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=8)

    # ── Header text ──────────────────────────────────────────────
    x0, y0 = rect.x + 8, rect.y + 6

    name_surf = fonts['name'].render(state_snap.name, True, (255, 255, 255))
    surface.blit(name_surf, (x0, y0))

    score_surf = fonts['stat'].render(f"Score: {score:,}", True, (255, 255, 255))
    surface.blit(score_surf, (x0, y0 + 26))

    if done:
        status_surf = fonts['stat'].render(f"DONE  {moves:,} moves", True, (255, 220, 80))
    else:
        mps_str = f"{mps:.0f} moves/sec" if mps >= 1 else f"{mps:.2f} moves/sec"
        status_surf = fonts['stat'].render(mps_str, True, (200, 240, 200))
    surface.blit(status_surf, (x0, y0 + 48))

    # ── Board tiles ───────────────────────────────────────────────
    board_x = rect.x + CELL_PAD
    board_y = rect.y + HEADER_H

    for r in range(grid):
        for c in range(grid):
            val  = int(board[r, c])
            tx   = board_x + c * (CELL_SIZE + CELL_PAD)
            ty   = board_y + r * (CELL_SIZE + CELL_PAD)
            tile = pygame.Rect(tx, ty, CELL_SIZE, CELL_SIZE)

            color = TILE_COLORS.get(val, DEFAULT_TILE_CLR)
            pygame.draw.rect(surface, color, tile, border_radius=5)

            if val != 0:
                tc  = (119, 110, 101) if val <= 4 else (255, 255, 255)
                num = fonts['tile'].render(str(val), True, tc)
                surface.blit(num, num.get_rect(center=tile.center))

    # ── Done overlay ─────────────────────────────────────────────
    if done:
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        surface.blit(overlay, rect.topleft)
        go_surf = fonts['gameover'].render("GAME OVER", True, (255, 80, 80))
        surface.blit(go_surf, go_surf.get_rect(center=rect.center))


# ─── Main entry point ─────────────────────────────────────────────

def run_multi_visual(agents, config):
    """
    Launch all agents in background threads and render them in one window.
    Exits when all games finish or Q is pressed.
    """
    pygame.init()
    grid_size = config.get('grid_size', 4)

    # Compute panel size from board dimensions
    board_px    = grid_size * (CELL_SIZE + CELL_PAD) + CELL_PAD
    panel_w     = board_px
    panel_h     = HEADER_H + board_px

    win_w = COLS * panel_w + (COLS + 1) * PANEL_PAD
    win_h = ROWS * panel_h + (ROWS + 1) * PANEL_PAD

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("2048 — Agent Race")
    clock  = pygame.time.Clock()

    fonts = {
        'name':     pygame.font.Font(None, 26),
        'stat':     pygame.font.Font(None, 22),
        'tile':     pygame.font.Font(None, max(18, 46 - grid_size * 4)),
        'gameover': pygame.font.Font(None, 40),
    }

    # Build per-agent state objects
    states = [AgentState(agent.name, grid_size) for agent in agents]

    # Compute panel rects (row-major)
    panel_rects = []
    for idx in range(len(agents)):
        row = idx // COLS
        col = idx % COLS
        rx  = PANEL_PAD + col * (panel_w + PANEL_PAD)
        ry  = PANEL_PAD + row * (panel_h + PANEL_PAD)
        panel_rects.append(pygame.Rect(rx, ry, panel_w, panel_h))

    # Start one thread per agent
    stop_event = threading.Event()
    threads = []
    for agent, state in zip(agents, states):
        t = threading.Thread(
            target=agent_loop,
            args=(agent, config, state, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print("Running — Q to quit early\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Stop when every agent has finished
        all_done = all(s.done for s in states)
        if all_done:
            # One final render then exit
            screen.fill(BG_COLOR)
            for state, rect in zip(states, panel_rects):
                with state.lock:
                    snap = copy.copy(state)
                    snap.board = state.board.copy()
                draw_panel(screen, snap, rect, fonts)
            pygame.display.flip()
            pygame.time.delay(1500)   # brief pause so user can see final state
            running = False

        screen.fill(BG_COLOR)
        for state, rect in zip(states, panel_rects):
            with state.lock:
                snap       = copy.copy(state)
                snap.board = state.board.copy()
            draw_panel(screen, snap, rect, fonts)

        pygame.display.flip()
        clock.tick(30)

    stop_event.set()
    pygame.quit()

    # ── Final results ─────────────────────────────────────────────
    print(f"\n{'Agent':<30} {'Score':>10}  {'Moves':>8}  {'Status'}")
    print("-" * 60)
    for state in states:
        status = "finished" if state.done else "interrupted"
        print(f"{state.name:<30} {state.score:>10,}  {state.moves:>8,}  {status}")


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = {'grid_size': 4, 'tile_2_probability': 0.9, 'initial_tiles': 2}
    agents = _build_agents()
    run_multi_visual(agents, config)
