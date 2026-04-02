"""
eval_pretrained.py
==================

Evaluates the pretrained AlphaZero network as a pure policy agent
(no MCTS — greedy argmax of policy head only).

This measures what imitation learning alone achieved, before any
AlphaZero self-play. Run in parallel while AlphaZero trains.

Usage::

    python agents/eval_pretrained.py --model models/alphazero_pretrained.pt --n_games 20
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from game.engine import Game2048, Action
from agents.alphazero import AlphaZeroNetwork, encode_board, _ACTION_LIST


def play_greedy(model: AlphaZeroNetwork, n_games: int = 20) -> None:
    """Play games using pure greedy policy (no MCTS)."""
    model.eval()
    config = {"grid_size": 4}

    scores, tiles, moves_list = [], [], []
    tile_counts = {}

    print(f"Running {n_games} games with greedy policy (no MCTS)...")
    print("-" * 60)
    t0 = time.time()

    for g in range(n_games):
        game  = Game2048(config)
        moves = 0

        while not game.is_game_over():
            board     = game.get_state()
            available = game.get_available_moves()

            # Get policy priors from network
            probs = model.predict_pi(board)  # shape (4,)

            # Mask illegal moves and pick greedy best legal action
            best_action = None
            best_prob   = -1.0
            for action in available:
                idx = _ACTION_LIST.index(action)
                if probs[idx] > best_prob:
                    best_prob   = probs[idx]
                    best_action = action

            game.move(best_action)
            moves += 1

        score    = game.get_score()
        max_tile = int(game.get_state().max())
        scores.append(score)
        tiles.append(max_tile)
        moves_list.append(moves)
        tile_counts[max_tile] = tile_counts.get(max_tile, 0) + 1

        print(f"  Game {g+1:>3}/{n_games} | Score: {score:>7} | "
              f"Max tile: {max_tile:>5} | Moves: {moves:>5}")

    elapsed = time.time() - t0

    print("-" * 60)
    print(f"\nResults over {n_games} games ({elapsed:.0f}s):")
    print(f"  Avg score:  {np.mean(scores):>8.0f}")
    print(f"  Med score:  {np.median(scores):>8.0f}")
    print(f"  Best score: {np.max(scores):>8.0f}")
    print(f"  Avg moves:  {np.mean(moves_list):>8.0f}")
    print(f"\nMax tile distribution:")
    for tile in sorted(tile_counts):
        bar   = "#" * tile_counts[tile]
        pct   = tile_counts[tile] / n_games * 100
        print(f"  {tile:>6}: {bar:<20} {tile_counts[tile]:>2}x  ({pct:.0f}%)")

    win_rate = sum(1 for t in tiles if t >= 2048) / n_games * 100
    r512     = sum(1 for t in tiles if t >= 512)  / n_games * 100
    print(f"\n  Reached  512+: {r512:.0f}%")
    print(f"  Reached 2048+: {win_rate:.0f}%")
    print(f"\nFor comparison:")
    print(f"  AlphaZero (self-play, no pretrain): avg ~1200-1600, max tile 256")
    print(f"  ExpectimaxSnake (depth=1):          avg ~30000,     max tile 2048+")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="models/alphazero_pretrained.pt")
    parser.add_argument("--n_games", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    model = AlphaZeroNetwork()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    print(f"Loaded: {args.model}")

    play_greedy(model, args.n_games)
