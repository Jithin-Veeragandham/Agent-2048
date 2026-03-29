"""
rollout.py
==========
Compares two greedy rollout strategies:
  - Clone approach:        sim.clone() per candidate move (current behaviour)
  - Save/restore approach: board.copy() + in-place move + restore

Runs N rollouts from a fixed board state and reports mean time per rollout.
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game import Game2048, Action
from agents.mcts import _default_heuristic

# ── Config ────────────────────────────────────────────────────────
ROLLOUT_DEPTH  = 15
NUM_ROLLOUTS   = 500
GRID_SIZE      = 4

# ── Hardcoded realistic mid-game board ───────────────────────────
game = Game2048.__new__(Game2048)
game.grid_size = GRID_SIZE
game.tile_2_prob = 0.9
game.initial_tiles = 2
game.seed = None
game.game_over = False
game.score = 8000
game.board = np.array([
    [512, 256, 128,  64],
    [ 32,  16,   8,   4],
    [  2,   4,   8,  16],
    [  0,   2,   4,   8],
], dtype=np.int32)

print(f"Benchmark board (after 40 random moves):")
print(game.board)
print(f"\nRollout depth : {ROLLOUT_DEPTH}")
print(f"Num rollouts  : {NUM_ROLLOUTS}")
print("-" * 50)


# ── Approach 1: Clone per candidate ──────────────────────────────
def rollout_clone(start_game):
    sim = start_game.clone()
    for _ in range(ROLLOUT_DEPTH):
        if sim.is_game_over():
            break
        moves = sim.get_available_moves()
        if not moves:
            break
        best_h = -float('inf')
        best_move = moves[0]
        for move in moves:
            child = sim.clone()
            child.move_fast(move)
            h = _default_heuristic(child.board)
            if h > best_h:
                best_h = h
                best_move = move
        sim.move_fast(best_move)
    return sim.get_score()


# ── Approach 2: Save/restore board ───────────────────────────────
def rollout_save_restore(start_game):
    sim = start_game.clone()
    for _ in range(ROLLOUT_DEPTH):
        if sim.is_game_over():
            break
        moves = sim.get_available_moves()
        if not moves:
            break
        best_h = -float('inf')
        best_move = moves[0]
        for move in moves:
            saved_board = sim.board.copy()
            saved_score = sim.score
            sim.move_fast(move)
            h = _default_heuristic(sim.board)
            if h > best_h:
                best_h = h
                best_move = move
            sim.board = saved_board
            sim.score = saved_score
        sim.move_fast(best_move)
    return sim.get_score()


# ── Warmup ────────────────────────────────────────────────────────
for _ in range(10):
    rollout_clone(game)
    rollout_save_restore(game)

# ── Benchmark clone ───────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(NUM_ROLLOUTS):
    rollout_clone(game)
clone_time = time.perf_counter() - t0

# ── Benchmark save/restore ────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(NUM_ROLLOUTS):
    rollout_save_restore(game)
sr_time = time.perf_counter() - t0

# ── Results ───────────────────────────────────────────────────────
print(f"\nClone approach      : {clone_time:.3f}s total  |  {clone_time/NUM_ROLLOUTS*1000:.2f} ms/rollout")
print(f"Save/restore approach: {sr_time:.3f}s total  |  {sr_time/NUM_ROLLOUTS*1000:.2f} ms/rollout")
print(f"\nSpeedup: {clone_time/sr_time:.2f}x  ({(1 - sr_time/clone_time)*100:.1f}% faster)")
