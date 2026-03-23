"""
tree_reuse.py
=============
Compares two MCTS tree strategies over a full game:
  - Fresh tree:  rebuild root from scratch every move (current behaviour)
  - Tree reuse:  re-root at the chosen child, carrying over visit counts

Measures total decision time across all moves in N games.
"""

import time
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game import Game2048, Action
from agents.mcts import _default_heuristic, _tree_value

# ── Config ────────────────────────────────────────────────────────
NUM_SIMULATIONS = 50       # per move (kept low so benchmark finishes fast)
ROLLOUT_DEPTH   = 5
EXPLORATION     = math.sqrt(2)
NUM_GAMES       = 5
GRID_SIZE       = 4


# ── Minimal MCTSNode (no imports from mcts to keep it self-contained) ─
class Node:
    __slots__ = ('game', 'parent', 'action', 'children', 'visits', 'value', 'untried', '_terminal')

    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried = game.get_available_moves()
        self._terminal = game.is_game_over() or len(self.untried) == 0

    @property
    def is_fully_expanded(self): return len(self.untried) == 0

    @property
    def is_terminal(self): return self._terminal

    def ucb1(self, c):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c):
        return max(self.children.values(), key=lambda ch: ch.ucb1(c))

    def expand(self):
        action = self.untried.pop()
        child_game = self.game.clone()
        child_game.move_fast(action)
        child = Node(child_game, parent=self, action=action)
        self.children[action] = child
        return child


def select(node):
    while not node.is_terminal and node.is_fully_expanded:
        node = node.best_child(EXPLORATION)
    return node


def rollout(game):
    sim = game.clone()
    for _ in range(ROLLOUT_DEPTH):
        if sim.is_game_over():
            break
        moves = sim.get_available_moves()
        if not moves:
            break
        best_h, best_move = -float('inf'), moves[0]
        for move in moves:
            child = sim.clone()
            child.move_fast(move)
            h = _default_heuristic(child.board)
            if h > best_h:
                best_h, best_move = h, move
        sim.move_fast(best_move)
    return _tree_value(sim)


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent


# ── Approach 1: Fresh tree every move ────────────────────────────
def play_game_fresh():
    game = Game2048({'grid_size': GRID_SIZE})
    total_decision_ms = 0.0

    while not game.is_game_over():
        moves = game.get_available_moves()
        if not moves:
            break
        if len(moves) == 1:
            game.move_fast(moves[0])
            continue

        t0 = time.perf_counter()
        root = Node(game.clone())          # fresh root every move
        for _ in range(NUM_SIMULATIONS):
            node = select(root)
            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand()
            value = rollout(node.game)
            backpropagate(node, value)

        best_action = max(root.children.values(), key=lambda c: c.visits).action
        total_decision_ms += (time.perf_counter() - t0) * 1000

        game.move_fast(best_action)

    return game.get_score(), total_decision_ms


# ── Approach 2: Tree reuse ────────────────────────────────────────
def play_game_reuse():
    game = Game2048({'grid_size': GRID_SIZE})
    total_decision_ms = 0.0
    root = Node(game.clone())              # build root once

    while not game.is_game_over():
        moves = game.get_available_moves()
        if not moves:
            break
        if len(moves) == 1:
            action = moves[0]
            game.move_fast(action)
            # re-root if child exists, else fresh
            if action in root.children:
                root = root.children[action]
                root.parent = None
            else:
                root = Node(game.clone())
            continue

        t0 = time.perf_counter()
        for _ in range(NUM_SIMULATIONS):
            node = select(root)
            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand()
            value = rollout(node.game)
            backpropagate(node, value)

        best_child = max(root.children.values(), key=lambda c: c.visits)
        best_action = best_child.action
        total_decision_ms += (time.perf_counter() - t0) * 1000

        game.move_fast(best_action)

        # Re-root at chosen child — carry over its subtree
        root = best_child
        root.parent = None

    return game.get_score(), total_decision_ms


# ── Run benchmark ─────────────────────────────────────────────────
print(f"MCTS Tree Reuse Benchmark")
print(f"  simulations/move : {NUM_SIMULATIONS}")
print(f"  rollout depth    : {ROLLOUT_DEPTH}")
print(f"  games            : {NUM_GAMES}")
print(f"  grid             : {GRID_SIZE}x{GRID_SIZE}")
print("-" * 50)

# Warmup
play_game_fresh()
play_game_reuse()

fresh_times, fresh_scores = [], []
reuse_times, reuse_scores = [], []

for i in range(NUM_GAMES):
    score, ms = play_game_fresh()
    fresh_times.append(ms)
    fresh_scores.append(score)
    print(f"  [fresh]  game {i+1}: score={score:6d}  decision_time={ms:.0f}ms")

print()

for i in range(NUM_GAMES):
    score, ms = play_game_reuse()
    reuse_times.append(ms)
    reuse_scores.append(score)
    print(f"  [reuse]  game {i+1}: score={score:6d}  decision_time={ms:.0f}ms")

print("\n" + "=" * 50)
print(f"Fresh tree  — avg decision time: {np.mean(fresh_times):.0f}ms  |  avg score: {np.mean(fresh_scores):.0f}")
print(f"Tree reuse  — avg decision time: {np.mean(reuse_times):.0f}ms  |  avg score: {np.mean(reuse_scores):.0f}")
print(f"\nTime speedup : {np.mean(fresh_times)/np.mean(reuse_times):.2f}x  ({(1 - np.mean(reuse_times)/np.mean(fresh_times))*100:.1f}% faster)")
print(f"Score change : {(np.mean(reuse_scores)/np.mean(fresh_scores) - 1)*100:+.1f}%")
