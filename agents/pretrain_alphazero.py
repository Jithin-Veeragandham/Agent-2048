"""
pretrain_alphazero.py
=====================

Imitation learning pre-trainer for AlphaZero.

Generates games from ExpectimaxSnakeAgent (a strong handcrafted agent),
records (board, action, score) at every step, then trains the AlphaZero
network to imitate those decisions.

Data format per game step:
    board     -> encode_board() -> (16, 4, 4) one-hot float32
                 channel i = 1 where log2(tile) == i  (ch0 = empty)
    action    -> int 0-3  (UP/DOWN/LEFT/RIGHT, same order as _ACTION_LIST)
    v_target  -> discounted MC return from this step forward, clipped [-1,1]
                 = sum_t gamma^t * (score_delta_t / _V_SCALE)

Supports parallel data collection and dataset caching so you only
run the slow ExpectimaxSnake search once.

Usage::

    # Collect 100 games using 6 parallel workers, cache dataset, train
    python agents/pretrain_alphazero.py --n_games 100 --depth 1 --n_workers 6

    # Reuse cached dataset for more training epochs (skip collection)
    python agents/pretrain_alphazero.py --load_data data/pretrain_dataset.npz --epochs 30

    # Then start AlphaZero with pre-trained weights
    python agents/alphazero.py --mode train --n_ep 500 --n_mcts 100 --fresh --pretrain_model models/alphazero_pretrained.pt
"""

import math
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional
import multiprocessing as mp
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import torch.optim as optim

from game.engine import Game2048, Action
from agents.expectimax_snake import ExpectimaxSnakeAgent
from agents.alphazero import AlphaZeroNetwork, encode_board, _ACTION_LIST, _V_SCALE

_NUM_ACTIONS = 4
_GAMMA = 0.99


# ═══════════════════════════════════════════════════════════════════
#  SINGLE-GAME WORKER  (top-level so multiprocessing can pickle it)
# ═══════════════════════════════════════════════════════════════════

def _run_one_game(depth: int) -> Tuple[List, int, int]:
    """Play one full game with ExpectimaxSnakeAgent and return trajectory.

    Returns:
        traj:        list of (board, action_idx, score_before_move)
        final_score: game score at termination
        max_tile:    highest tile reached
    """
    agent = ExpectimaxSnakeAgent(depth=depth)
    game  = Game2048({"grid_size": 4})
    traj  = []

    while not game.is_game_over():
        board     = game.get_state()
        score     = game.get_score()
        available = game.get_available_moves()
        action    = agent.choose_action(board, available)
        action_idx = _ACTION_LIST.index(action)
        traj.append((board.copy(), action_idx, score))
        game.move(action)

    final_score = game.get_score()
    max_tile    = int(game.get_state().max())
    return traj, final_score, max_tile


# ═══════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_games(n_games: int, depth: int = 1, n_workers: int = 1) -> List[Tuple]:
    """Run ExpectimaxSnake games and record (board, action_idx, score) per step.

    Args:
        n_games:   Number of complete games to collect.
        depth:     ExpectimaxSnake search depth (1 = fast, 2 = strong).
        n_workers: Parallel worker processes. >1 uses multiprocessing.Pool.

    Returns:
        List of (trajectory, final_score) tuples.
    """
    print(f"Collecting {n_games} games | depth={depth} | workers={n_workers}")
    t0 = time.time()

    if n_workers > 1:
        ctx  = mp.get_context("spawn")
        args = [depth] * n_games
        with ctx.Pool(processes=n_workers) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_run_one_game, args), 1):
                results.append(res)
                if i % max(1, n_games // 10) == 0 or i == n_games:
                    elapsed  = time.time() - t0
                    avg_sc   = np.mean([r[1] for r in results])
                    avg_tile = np.mean([r[2] for r in results])
                    print(f"  {i:>4}/{n_games} done | "
                          f"Avg score: {avg_sc:>7.0f} | "
                          f"Avg tile: {avg_tile:>5.0f} | "
                          f"{elapsed:.0f}s elapsed")
    else:
        results = []
        for g in range(n_games):
            res = _run_one_game(depth)
            results.append(res)
            if (g + 1) % max(1, n_games // 10) == 0 or g == 0:
                elapsed = time.time() - t0
                avg_sc  = np.mean([r[1] for r in results])
                print(f"  Game {g+1:>4}/{n_games} | "
                      f"Score: {res[1]:>6} | Tile: {res[2]:>4} | "
                      f"Avg: {avg_sc:>6.0f} | {elapsed:.0f}s elapsed")

    trajectories = [(r[0], r[1]) for r in results]
    avg   = np.mean([r[1] for r in results])
    best  = max(r[1] for r in results)
    tiles = sorted(set(r[2] for r in results))
    print(f"\nCollection done in {time.time()-t0:.0f}s | "
          f"Avg: {avg:.0f} | Best: {best} | Tiles seen: {tiles}")
    return trajectories


# ═══════════════════════════════════════════════════════════════════
#  DATASET BUILD / SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

def build_dataset(trajectories: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert trajectories into flat arrays for training.

    Returns:
        boards:    (N, 16, 4, 4) float32  — one-hot encoded board states
        actions:   (N,)          int64    — expert action index per step
        v_targets: (N,)          float32  — MC returns normalised to [-1, 1]
    """
    boards_list, actions_list, values_list = [], [], []

    for traj, final_score in trajectories:
        all_scores = [s for _, _, s in traj] + [final_score]
        G = 0.0
        returns = []
        for t in range(len(traj) - 1, -1, -1):
            r_t = (all_scores[t + 1] - all_scores[t]) / _V_SCALE
            G   = r_t + _GAMMA * G
            returns.append(max(-1.0, min(1.0, G)))
        returns.reverse()

        for i, (board, action_idx, _) in enumerate(traj):
            boards_list.append(encode_board(board))
            actions_list.append(action_idx)
            values_list.append(returns[i])

    boards  = np.stack(boards_list).astype(np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    values  = np.array(values_list,  dtype=np.float32)

    print(f"Dataset: {len(boards):,} steps | "
          f"Action dist: {np.bincount(actions, minlength=4)} | "
          f"V range: [{values.min():.3f}, {values.max():.3f}]")
    return boards, actions, values


def save_dataset(path: str, boards: np.ndarray, actions: np.ndarray, values: np.ndarray):
    """Save dataset arrays to a .npz file for reuse."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, boards=boards, actions=actions, values=values)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Dataset saved to {path}  ({size_mb:.1f} MB)")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a previously saved dataset .npz file."""
    data = np.load(path)
    boards  = data["boards"]
    actions = data["actions"]
    values  = data["values"]
    print(f"Loaded dataset from {path}: {len(boards):,} steps | "
          f"Action dist: {np.bincount(actions, minlength=4)} | "
          f"V range: [{values.min():.3f}, {values.max():.3f}]")
    return boards, actions, values


# ═══════════════════════════════════════════════════════════════════
#  PRE-TRAINING
# ═══════════════════════════════════════════════════════════════════

def pretrain(
    boards: np.ndarray,
    actions: np.ndarray,
    v_targets: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.001,
    save_path: str = "models/alphazero_pretrained.pt",
):
    """Train AlphaZero network on Expectimax demonstrations.

    Policy head: cross-entropy against expert actions.
    Value head:  MSE against MC returns from expert games.
    """
    model     = AlphaZeroNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    N        = len(boards)
    n_batches = math.ceil(N / batch_size)

    boards_t  = torch.from_numpy(boards)
    actions_t = torch.from_numpy(actions)
    values_t  = torch.from_numpy(v_targets).unsqueeze(1)

    print(f"\nPre-training: {epochs} epochs | batch={batch_size} | lr={lr} | {N:,} samples")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        perm      = torch.randperm(N)
        boards_t  = boards_t[perm]
        actions_t = actions_t[perm]
        values_t  = values_t[perm]

        epoch_pi_loss = 0.0
        epoch_v_loss  = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, N)

            log_pi, v_pred = model(boards_t[start:end])

            pi_loss = F.nll_loss(log_pi, actions_t[start:end])
            v_loss  = F.mse_loss(v_pred, values_t[start:end])
            loss    = pi_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_pi_loss += pi_loss.item()
            epoch_v_loss  += v_loss.item()

        avg_pi = epoch_pi_loss / n_batches
        avg_v  = epoch_v_loss  / n_batches
        total  = avg_pi + avg_v

        model.eval()
        with torch.no_grad():
            log_pi_all, _ = model(boards_t)
            acc = (log_pi_all.argmax(dim=1) == actions_t).float().mean().item()

        marker = " *" if total < best_loss else ""
        print(f"  Epoch {epoch+1:>3}/{epochs} | "
              f"pi_loss: {avg_pi:.4f} | v_loss: {avg_v:.4f} | "
              f"total: {total:.4f} | policy_acc: {acc:.1%}{marker}")

        if total < best_loss:
            best_loss = total
            torch.save(model.state_dict(), save_path)

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    print(f"Saved to: {save_path}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train AlphaZero from Expectimax")
    parser.add_argument("--n_games",    type=int,   default=100,
                        help="Number of Expectimax games to collect (ignored if --load_data)")
    parser.add_argument("--depth",      type=int,   default=1,
                        help="ExpectimaxSnake search depth (1=fast ~55s/game, 2=strong ~25min/game)")
    parser.add_argument("--n_workers",  type=int,   default=1,
                        help="Parallel worker processes for game collection")
    parser.add_argument("--save_data",  type=str,   default=None,
                        help="Save collected dataset to this .npz path for reuse")
    parser.add_argument("--load_data",  type=str,   default=None,
                        help="Load a previously saved dataset .npz instead of collecting games")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--save",       type=str,   default="models/alphazero_pretrained.pt",
                        help="Output model path")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # 1. Get dataset (load cached or collect fresh)
    if args.load_data:
        boards, actions, v_targets = load_dataset(args.load_data)
    else:
        trajectories = collect_games(args.n_games, args.depth, args.n_workers)
        boards, actions, v_targets = build_dataset(trajectories)
        if args.save_data:
            save_dataset(args.save_data, boards, actions, v_targets)

    # 2. Pre-train
    pretrain(
        boards, actions, v_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save,
    )

    print(f"\nTo use pre-trained weights:")
    print(f"  python agents/alphazero.py --mode train --n_ep 500 "
          f"--n_mcts 100 --fresh --pretrain_model {args.save}")
