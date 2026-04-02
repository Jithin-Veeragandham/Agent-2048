"""
alphazero.py
============

AlphaZero agent for 2048, based on tmoer.github.io/AlphaZero.

Combines Monte Carlo Tree Search with a neural network that provides
both a policy prior (action probabilities) and a value estimate
(expected future return). The network is trained via self-play:
MCTS produces improved policy targets, and actual game returns
provide stable value targets (Monte Carlo returns).

Adaptations for 2048 (stochastic single-player game):
    - Chance node sampling: at each MCTS leaf, N tile placements are
      sampled and the network value is averaged, giving a stable
      expectation over random tile spawns.
    - Monte Carlo returns: value targets use actual discounted game
      returns instead of bootstrapped Q-values, removing cold-start
      noise from the training signal.
    - No subtree reuse: random tile spawns make subtrees stale.
    - Legal action masking throughout MCTS and inference.

Usage::

    # Training (fresh start)
    python agents/alphazero.py --mode train --n_ep 500 --n_mcts 100 --fresh

    # Resume from checkpoint
    python agents/alphazero.py --mode train --n_ep 500 --n_mcts 100

    # Evaluation
    python agents/alphazero.py --mode eval --model models/alphazero_best.pt --num_games 10

Reference: tmoer.github.io/AlphaZero (Thomas Moerland, 2017)
"""

import math
import random
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game.engine import Game2048, Action
from agents.base import BaseAgent


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

_ACTION_LIST = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
_NUM_ACTIONS = 4
_NUM_TILE_CHANNELS = 16   # exponents 0-15 (tile values 0, 2, 4, ..., 32768)
_V_SCALE = 30000.0        # normalise MC returns to roughly [-1, 1]

# This model is bottlenecked by sequential MCTS Python calls, not matrix math.
# GPU transfer overhead on tiny (1,16,4,4) tensors exceeds compute savings.
# Everything runs on CPU; DEVICE kept for reporting only.
DEVICE = torch.device("cpu")
INFER_DEVICE = DEVICE
TRAIN_DEVICE = DEVICE


# ═══════════════════════════════════════════════════════════════════
#  FAST NUMPY BOARD OPS (move without spawning a tile)
# ═══════════════════════════════════════════════════════════════════

def _slide_row_left_np(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """Slide and merge a 4-element row leftward. Returns (new_row, score)."""
    tiles = row[row != 0]
    result = np.zeros(4, dtype=np.int64)
    score = 0
    i = j = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = int(tiles[i] * 2)
            result[j] = merged
            score += merged
            i += 2
        else:
            result[j] = tiles[i]
            i += 1
        j += 1
    return result, score


def _apply_move_np(board: np.ndarray, action: Action) -> Tuple[np.ndarray, bool, int]:
    """Apply a move WITHOUT spawning a tile.

    Returns:
        new_board: (4,4) int64 post-move board
        changed:   whether the board changed
        score_delta: points earned by merges
    """
    b = board.astype(np.int64)
    new_board = np.empty((4, 4), dtype=np.int64)
    score = 0

    if action == Action.LEFT:
        for r in range(4):
            new_board[r], s = _slide_row_left_np(b[r])
            score += s
    elif action == Action.RIGHT:
        for r in range(4):
            rev, s = _slide_row_left_np(b[r, ::-1])
            new_board[r] = rev[::-1]
            score += s
    elif action == Action.UP:
        for c in range(4):
            col, s = _slide_row_left_np(b[:, c])
            new_board[:, c] = col
            score += s
    else:  # DOWN
        for c in range(4):
            col, s = _slide_row_left_np(b[::-1, c])
            new_board[::-1, c] = col
            score += s

    changed = not np.array_equal(b, new_board)
    return new_board, changed, score


# ═══════════════════════════════════════════════════════════════════
#  BOARD ENCODING
# ═══════════════════════════════════════════════════════════════════

def encode_board(board: np.ndarray) -> np.ndarray:
    """One-hot encode tile exponents into (16, 4, 4) float32 array.

    Channel i is 1 where log2(tile) == i (channel 0 = empty cells).
    """
    encoded = np.zeros((_NUM_TILE_CHANNELS, 4, 4), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            if val == 0:
                encoded[0, r, c] = 1.0
            else:
                exp = int(math.log2(val))
                if exp < _NUM_TILE_CHANNELS:
                    encoded[exp, r, c] = 1.0
    return encoded


def encode_board_batch(boards: List[np.ndarray]) -> torch.Tensor:
    """Encode a batch of boards into a (B, 16, 4, 4) tensor."""
    return torch.from_numpy(np.stack([encode_board(b) for b in boards]))


# ═══════════════════════════════════════════════════════════════════
#  NEURAL NETWORK (Policy + Value heads)
# ═══════════════════════════════════════════════════════════════════

class AlphaZeroNetwork(nn.Module):
    """CNN with shared backbone, policy head, and value head.

    Architecture:
        Input:  (B, 16, 4, 4) one-hot encoded board
        Conv:   128 filters 3x3 (pad=1) -> BN -> ReLU
                128 filters 3x3 (pad=1) -> BN -> ReLU
        Flatten -> FC(2048, 256) -> ReLU
        Policy: FC(256, 4) -> log_softmax
        Value:  FC(256, 1) -> tanh (normalised to ~[-1, 1])
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(_NUM_TILE_CHANNELS, 128, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.fc    = nn.Linear(128 * 4 * 4, 256)
        self.policy_head = nn.Linear(256, _NUM_ACTIONS)
        self.value_head  = nn.Linear(256, 1)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_policy, value)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        log_pi = F.log_softmax(self.policy_head(x), dim=1)
        v = torch.tanh(self.value_head(x))
        return log_pi, v

    def predict_pi(self, board: np.ndarray) -> np.ndarray:
        """Single board -> policy priors (numpy float32)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(encode_board(board)).unsqueeze(0)
            log_pi, _ = self.forward(x)
            return torch.exp(log_pi).squeeze(0).numpy()

    def predict_v(self, board: np.ndarray) -> float:
        """Single board -> value estimate (float in [-1, 1])."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(encode_board(board)).unsqueeze(0)
            _, v = self.forward(x)
            return v.item()

    def predict_pi_batch(self, boards: np.ndarray) -> np.ndarray:
        """Batch of boards (N,4,4) -> policy priors (N,4) numpy."""
        self.eval()
        with torch.no_grad():
            encoded = np.stack([encode_board(b) for b in boards])
            x = torch.from_numpy(encoded)
            log_pi, _ = self.forward(x)
            return torch.exp(log_pi).numpy()

    def predict_v_batch(self, boards: np.ndarray) -> np.ndarray:
        """Batch of boards (N,4,4) -> values (N,) numpy."""
        self.eval()
        with torch.no_grad():
            encoded = np.stack([encode_board(b) for b in boards])
            x = torch.from_numpy(encoded)
            _, v = self.forward(x)
            return v.squeeze(1).numpy()


# ═══════════════════════════════════════════════════════════════════
#  MCTS (AlphaZero-style with PUCT + chance node sampling)
# ═══════════════════════════════════════════════════════════════════

class MCTSNode:
    """A node in the MCTS tree representing a concrete game state."""

    def __init__(self, board: np.ndarray, score: int, game_over: bool,
                 parent_action_idx: Optional[int], model: AlphaZeroNetwork,
                 legal_mask: np.ndarray):
        self.board = board
        self.score = score
        self.game_over = game_over
        self.parent_action_idx = parent_action_idx
        self.n = 0

        self.legal_mask = legal_mask

        if not game_over:
            priors = model.predict_pi(board)
            self.value = model.predict_v(board)
            priors = priors * legal_mask
            s = priors.sum()
            self.priors = priors / s if s > 0 else legal_mask / legal_mask.sum()
        else:
            self.priors = np.zeros(_NUM_ACTIONS, dtype=np.float32)
            self.value = 0.0

        self.child_W = np.zeros(_NUM_ACTIONS, dtype=np.float64)
        self.child_n = np.zeros(_NUM_ACTIONS, dtype=np.int32)
        self.child_Q = np.zeros(_NUM_ACTIONS, dtype=np.float64)
        self.children: List[Optional['MCTSNode']] = [None] * _NUM_ACTIONS

    def select_action(self, c: float = 1.5) -> int:
        """PUCT selection: Q + prior * c * sqrt(N+1) / (n+1)."""
        puct = self.child_Q + self.priors * c * (
            math.sqrt(self.n + 1) / (self.child_n + 1)
        )
        puct = np.where(self.legal_mask, puct, -np.inf)
        max_val = puct.max()
        candidates = np.where(puct == max_val)[0]
        return int(np.random.choice(candidates))

    def update_child(self, action_idx: int, value: float):
        self.child_n[action_idx] += 1
        self.child_W[action_idx] += value
        self.child_Q[action_idx] = self.child_W[action_idx] / self.child_n[action_idx]
        self.n += 1


def _chance_leaf_value(post_move_board: np.ndarray, score_delta: int,
                       model: AlphaZeroNetwork, n_chance: int) -> float:
    """Estimate leaf value by averaging over N random tile placements.

    This approximates the expectation over the chance node:
        E[V(s')] = sum_{empty cells} P(cell) * [0.9*V(cell+2) + 0.1*V(cell+4)]

    Using Monte Carlo sampling with n_chance samples.
    """
    empty_cells = np.argwhere(post_move_board == 0)
    if len(empty_cells) == 0:
        # Board full — no tile spawn possible
        child_value = model.predict_v(post_move_board)
        return score_delta / _V_SCALE + 0.99 * child_value

    n_samples = min(n_chance, len(empty_cells))
    sample_boards = []
    for _ in range(n_samples):
        idx = np.random.randint(len(empty_cells))
        r, c = empty_cells[idx]
        tile = 2 if np.random.random() < 0.9 else 4
        child = post_move_board.copy()
        child[r, c] = tile
        sample_boards.append(child)

    # Batch network call for efficiency
    child_values = model.predict_v_batch(np.stack(sample_boards))
    avg_child_value = float(child_values.mean())
    return score_delta / _V_SCALE + 0.99 * avg_child_value


def run_mcts(root_board: np.ndarray, root_score: int,
             model: AlphaZeroNetwork, n_mcts: int, c: float,
             config: Dict, n_chance: int = 3) -> Tuple[np.ndarray, float]:
    """Run MCTS simulations and return (policy_target, value_target).

    Key improvement over naive MCTS:
        At each leaf expansion, we apply the player move using fast numpy
        ops (no tile spawn), then sample n_chance random tile placements
        and average the network's value estimates. This gives a stable
        expectation over chance outcomes rather than a single noisy sample.

    Returns:
        pi: (4,) visit counts normalised to a probability distribution
        v:  visit-weighted Q-value at the root (used only for diagnostics;
            training uses Monte Carlo returns instead)
    """
    game = Game2048.from_state(root_board, score=root_score, config=config)
    legal_moves = game.get_available_moves()
    legal_mask = np.array([a in legal_moves for a in _ACTION_LIST], dtype=np.float32)

    root = MCTSNode(root_board, root_score, game.is_game_over(),
                    None, model, legal_mask)

    if root.game_over:
        return np.ones(_NUM_ACTIONS) / _NUM_ACTIONS, 0.0

    for _ in range(n_mcts):
        node = root
        action_path = []

        # ── Selection: descend until unexpanded node ──────────
        while True:
            if node.game_over:
                leaf_value = 0.0
                break

            a = node.select_action(c)
            action_path.append((node, a))

            if node.children[a] is None:
                # ── Expansion ────────────────────────────────
                # Apply move using fast numpy (no tile spawn yet)
                post_board, changed, delta_score = _apply_move_np(
                    node.board, _ACTION_LIST[a]
                )

                if not changed:
                    # Illegal move slipped through legal_mask — penalise
                    leaf_value = -1.0
                    break

                # Chance node: sample n_chance tile outcomes, average V
                leaf_value = _chance_leaf_value(
                    post_board, delta_score, model, n_chance
                )

                # Build child node with ONE actual tile spawn (for tree)
                sim_game = Game2048.from_state(
                    node.board, score=node.score, config=config
                )
                valid, _ = sim_game.move(_ACTION_LIST[a])
                if not valid:
                    leaf_value = -1.0
                    break

                child_board = sim_game.get_state()
                child_score = sim_game.get_score()
                child_over  = sim_game.is_game_over()
                child_legal = (
                    np.array([act in sim_game.get_available_moves()
                               for act in _ACTION_LIST], dtype=np.float32)
                    if not child_over
                    else np.zeros(_NUM_ACTIONS, dtype=np.float32)
                )

                node.children[a] = MCTSNode(
                    child_board, child_score, child_over,
                    a, model, child_legal
                )
                break
            else:
                node = node.children[a]

        # ── Backpropagation ───────────────────────────────────
        for (ancestor, action_idx) in reversed(action_path):
            ancestor.update_child(action_idx, leaf_value)
            leaf_value = 0.99 * leaf_value  # discount as we go up

    # ── Extract policy target ─────────────────────────────────
    counts = root.child_n.astype(np.float64)
    count_sum = counts.sum()
    if count_sum > 0:
        pi_target = counts / count_sum
    else:
        pi_target = legal_mask / legal_mask.sum()

    # Q-value for diagnostics (not used as training target)
    v_target = float(np.sum((counts / count_sum) * root.child_Q)) if count_sum > 0 else 0.0

    return pi_target, v_target


# ═══════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size FIFO buffer for (board, pi_target, v_target) tuples."""

    def __init__(self, max_size: int = 50000):
        self.buffer = deque(maxlen=max_size)

    def store(self, board: np.ndarray, pi: np.ndarray, v: float):
        self.buffer.append((board, pi, v))

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        boards, pis, vs = [], [], []
        for i in indices:
            b, p, v = self.buffer[i]
            boards.append(encode_board(b))
            pis.append(p)
            vs.append(v)
        return (
            torch.from_numpy(np.stack(boards)),
            torch.from_numpy(np.stack(pis)),
            torch.tensor(vs, dtype=torch.float32).unsqueeze(1),
        )


# ═══════════════════════════════════════════════════════════════════
#  ALPHAZERO TRAINER
# ═══════════════════════════════════════════════════════════════════

class AlphaZeroTrainer:
    """AlphaZero training loop with improvements for single-player stochastic games:

    1. Chance node sampling: MCTS leaf values average over N tile placements.
    2. Monte Carlo returns: value targets use actual discounted game returns.
    3. Best-of-N self-play: play N games per episode, store only the best.
       Prevents bad games from polluting the replay buffer.
    4. Temperature schedule: linearly decay exploration from 1.0 → temp_final
       over temp_anneal_ep episodes, then stay greedy.
    5. Buffer score filter: reject trajectories below min_score_ratio * EMA
       of recent scores (with warmup period). Breaks negative feedback loop.
    """

    def __init__(
        self,
        n_ep: int = 500,
        n_mcts: int = 100,
        c: float = 1.5,
        n_chance: int = 3,
        lr: float = 0.001,
        batch_size: int = 64,
        buffer_size: int = 50000,
        save_dir: str = "models",
        eval_interval: int = 25,
        eval_games: int = 5,
        gamma: float = 0.99,
        n_games_per_ep: int = 2,
        temp_anneal_ep: int = 200,
        temp_final: float = 0.1,
        min_score_ratio: float = 0.7,
        fresh: bool = False,
        pretrain_model: Optional[str] = None,
    ):
        self.n_ep = n_ep
        self.n_mcts = n_mcts
        self.c = c
        self.n_chance = n_chance
        self.lr = lr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.gamma = gamma
        self.n_games_per_ep = n_games_per_ep
        self.temp_anneal_ep = temp_anneal_ep
        self.temp_final = temp_final
        self.min_score_ratio = min_score_ratio
        self.config = {"grid_size": 4}

        self.model = AlphaZeroNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = ReplayBuffer(max_size=buffer_size)

        os.makedirs(save_dir, exist_ok=True)

        self.best_avg_score = 0.0
        self.start_ep = 0
        self.score_ema: Optional[float] = None  # exponential moving avg of scores

        ckpt_path = os.path.join(save_dir, "alphazero_checkpoint.pt")
        if not fresh and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)
        elif fresh and pretrain_model and os.path.exists(pretrain_model):
            self.model.load_state_dict(
                torch.load(pretrain_model, map_location=DEVICE, weights_only=True)
            )
            print(f"Loaded pre-trained weights from {pretrain_model}")

    def _save_checkpoint(self, ep: int):
        path = os.path.join(self.save_dir, "alphazero_checkpoint.pt")
        torch.save({
            'episode': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_avg_score': self.best_avg_score,
            'score_ema': self.score_ema,
            'buffer': list(self.buffer.buffer),
        }, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_avg_score = ckpt.get('best_avg_score', 0.0)
        self.score_ema = ckpt.get('score_ema', None)
        self.start_ep = ckpt.get('episode', 0) + 1
        for item in ckpt.get('buffer', []):
            self.buffer.buffer.append(item)
        print(f"Resumed from episode {self.start_ep} "
              f"(buffer={len(self.buffer)}, best={self.best_avg_score:.0f})")

    def _play_one_game(self, temp: float) -> Tuple[List, float, int]:
        """Play one complete game. Returns (trajectory, final_score, max_tile).

        Does NOT write to the buffer — caller decides whether to store.
        trajectory: list of (board, pi_target, score_before_move)
        """
        game = Game2048(self.config)
        trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not game.is_game_over():
            board = game.get_state()
            score = game.get_score()

            pi_target, _ = run_mcts(
                board, score, self.model, self.n_mcts, self.c,
                self.config, self.n_chance
            )

            trajectory.append((board.copy(), pi_target, score))

            # Apply temperature, then sample action
            legal_moves = game.get_available_moves()
            legal_mask = np.array([a in legal_moves for a in _ACTION_LIST],
                                  dtype=np.float32)
            if temp != 1.0 and temp > 0:
                pi_t = pi_target ** (1.0 / temp)
            else:
                pi_t = pi_target.copy()
            pi_t = pi_t * legal_mask
            s = pi_t.sum()
            pi_t = pi_t / s if s > 0 else legal_mask / legal_mask.sum()

            # Greedy at very low temp
            if temp < 0.05:
                action_idx = int(np.argmax(pi_t))
            else:
                action_idx = int(np.random.choice(_NUM_ACTIONS, p=pi_t))
            action = _ACTION_LIST[action_idx]

            valid, _ = game.move(action)
            if not valid:
                game.move(legal_moves[0])

        final_score = float(game.get_score())
        max_tile = int(game.get_state().max())
        return trajectory, final_score, max_tile

    def _store_trajectory(self, trajectory: List, final_score: float):
        """Compute MC returns for a trajectory and store in replay buffer."""
        all_scores = [s for _, _, s in trajectory] + [final_score]
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            r_t = (all_scores[t + 1] - all_scores[t]) / _V_SCALE
            G = r_t + self.gamma * G
            v_target = max(-1.0, min(1.0, G))
            board_t, pi_t, _ = trajectory[t]
            self.buffer.store(board_t, pi_t, v_target)

    def self_play_episode(self, ep: int, temp: float) -> Tuple[float, int]:
        """Play n_games_per_ep games, store only the best-scoring one.

        Improvements applied here:
        - Best-of-N: run N games, keep the highest-scoring trajectory.
          Prevents bad games from polluting the replay buffer.
        - Buffer filter: skip storing if the best score is below
          min_score_ratio * EMA of recent scores (with warmup period).
          Breaks the negative feedback loop where bad games reinforce
          a bad policy.

        Returns (best_score, best_max_tile).
        """
        best_score = -1.0
        best_traj: Optional[List] = None
        best_tile = 0

        for _ in range(self.n_games_per_ep):
            traj, score, tile = self._play_one_game(temp)
            if score > best_score:
                best_score = score
                best_traj  = traj
                best_tile  = tile

        # Update score EMA
        if self.score_ema is None:
            self.score_ema = best_score
        else:
            self.score_ema = 0.9 * self.score_ema + 0.1 * best_score

        # Buffer filter: warmup for first 20 episodes, then apply threshold
        warmup = ep < 20
        above_threshold = best_score >= self.score_ema * self.min_score_ratio
        if warmup or above_threshold:
            self._store_trajectory(best_traj, best_score)

        return best_score, best_tile

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.model.train()
        boards, pi_targets, v_targets = self.buffer.sample_batch(self.batch_size)

        log_pi, v_pred = self.model(boards)

        pi_loss = -(pi_targets * log_pi).sum(dim=1).mean()
        v_loss  = F.mse_loss(v_pred, v_targets)
        loss    = pi_loss + v_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(self, num_games: int = 5) -> Tuple[float, float, int]:
        """Evaluate model greedily (no MCTS). Returns (avg_score, win_rate, max_tile)."""
        scores = []
        wins = 0
        best_tile = 0

        for _ in range(num_games):
            game = Game2048(self.config)
            while not game.is_game_over():
                board = game.get_state()
                pi = self.model.predict_pi(board)
                legal_moves = game.get_available_moves()
                legal_mask = np.array([a in legal_moves for a in _ACTION_LIST])
                pi = pi * legal_mask
                if pi.sum() > 0:
                    action_idx = int(np.argmax(pi))
                else:
                    action_idx = _ACTION_LIST.index(legal_moves[0])
                game.move(_ACTION_LIST[action_idx])

            score = game.get_score()
            tile  = int(game.get_state().max())
            scores.append(score)
            if tile >= 2048:
                wins += 1
            best_tile = max(best_tile, tile)

        return sum(scores) / len(scores), wins / num_games, best_tile

    def train(self):
        print(f"AlphaZero Training: {self.n_ep} episodes, "
              f"{self.n_mcts} MCTS sims/move, {self.n_chance} chance samples")
        print(f"Best-of-{self.n_games_per_ep} self-play | "
              f"Temp: 1.0 -> {self.temp_final} over {self.temp_anneal_ep} eps | "
              f"Buffer filter: {self.min_score_ratio:.0%} of EMA")
        print(f"Value targets: Monte Carlo returns (gamma={self.gamma})")
        print(f"Device: cpu (GPU skipped — MCTS bottleneck is Python loop, not matrix math)")
        if self.start_ep > 0:
            print(f"Resuming from episode {self.start_ep}")
        print("=" * 60)

        best_score_ever = 0
        best_tile_ever  = 0
        recent_scores   = deque(maxlen=25)
        total_wins = 0
        total_games = 0

        for ep in range(self.start_ep, self.n_ep):
            ep_start = time.time()

            # Temperature schedule: linear decay 1.0 → temp_final
            frac = min(1.0, ep / max(1, self.temp_anneal_ep))
            temp = 1.0 - frac * (1.0 - self.temp_final)

            score, max_tile = self.self_play_episode(ep, temp)

            best_score_ever = max(best_score_ever, score)
            best_tile_ever  = max(best_tile_ever, max_tile)
            recent_scores.append(score)
            total_games += 1
            if max_tile >= 2048:
                total_wins += 1
            avg_recent = sum(recent_scores) / len(recent_scores)

            # Multiple gradient steps per episode
            n_steps = min(max(1, len(self.buffer) // self.batch_size), 10)
            losses = [self.train_step() for _ in range(n_steps)]
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            ep_time = time.time() - ep_start
            eta_min = ep_time * (self.n_ep - ep - 1) / 60

            print(f"  Ep {ep+1:>4}/{self.n_ep} | Score: {score:>8.0f} | "
                  f"Max: {max_tile:>5} | Loss: {avg_loss:.4f} | "
                  f"Avg25: {avg_recent:>8.0f} | "
                  f"Best: {best_score_ever:>8.0f} ({best_tile_ever}) | "
                  f"T:{temp:.2f} | WinRate: {100*total_wins/total_games:.0f}% | "
                  f"{ep_time:.1f}s (ETA: {eta_min:.0f}min)")

            if (ep + 1) % self.eval_interval == 0:
                avg_score, win_rate, best_tile = self.evaluate(self.eval_games)
                print(f"  >>> Eval ({self.eval_games} games): "
                      f"Avg={avg_score:.0f}, WinRate={win_rate:.0%}, "
                      f"BestTile={best_tile}")

                if avg_score > self.best_avg_score:
                    self.best_avg_score = avg_score
                    path = os.path.join(self.save_dir, "alphazero_best.pt")
                    torch.save(self.model.state_dict(), path)
                    print(f"  >>> New best! Saved to {path}")

                self._save_checkpoint(ep)
                print(f"  >>> Checkpoint saved (ep {ep+1})")

        self._save_checkpoint(self.n_ep - 1)
        print(f"\nTraining complete.")
        print(f"  Best self-play score: {best_score_ever:.0f} (tile: {best_tile_ever})")
        print(f"  Self-play win rate: {100*total_wins/total_games:.1f}%")
        print(f"  Best eval avg score: {self.best_avg_score:.0f}")


# ═══════════════════════════════════════════════════════════════════
#  AGENT (framework integration)
# ═══════════════════════════════════════════════════════════════════

class AlphaZeroAgent(BaseAgent):
    """AlphaZero agent for 2048.

    Args:
        model_path: Path to trained .pt file.
        n_mcts: MCTS simulations per move (0 = greedy policy only).
        c: PUCT exploration constant.
        n_chance: Chance node samples per MCTS expansion.
    """

    agent_type = "alphazero"

    def __init__(self, model_path: Optional[str] = None,
                 n_mcts: int = 100, c: float = 1.5, n_chance: int = 3):
        mode = f"mcts={n_mcts}" if n_mcts > 0 else "greedy"
        super().__init__(f"AlphaZero({mode})")
        self.n_mcts = n_mcts
        self.c = c
        self.n_chance = n_chance
        self.config = {"grid_size": 4}

        self.model = AlphaZeroNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True)
            )
            print(f"Loaded AlphaZero model from {model_path}")
        self.model.eval()

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        if len(available_moves) == 1:
            return available_moves[0]

        board = state
        score = game_context.get('score', 0) if game_context else 0
        legal_mask = np.array([a in available_moves for a in _ACTION_LIST])

        if self.n_mcts > 0:
            pi, _ = run_mcts(board, score, self.model, self.n_mcts, self.c,
                             self.config, self.n_chance)
        else:
            pi = self.model.predict_pi(board)

        pi = pi * legal_mask
        if pi.sum() > 0:
            action_idx = int(np.argmax(pi))
        else:
            return available_moves[0]

        return _ACTION_LIST[action_idx]

    def get_params(self) -> Dict:
        return {'n_mcts': self.n_mcts, 'c': self.c, 'n_chance': self.n_chance}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero 2048")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--n_ep", type=int, default=500)
    parser.add_argument("--n_mcts", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--n_chance", type=int, default=3,
                        help="Tile placements to sample per MCTS expansion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--eval_games", type=int, default=5)
    parser.add_argument("--model", type=str, default="models/alphazero_best.pt")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--n_games_per_ep", type=int, default=2,
                        help="Games per episode (best-of-N stored)")
    parser.add_argument("--temp_anneal_ep", type=int, default=200,
                        help="Episodes over which to decay temperature 1.0->temp_final")
    parser.add_argument("--temp_final", type=float, default=0.1,
                        help="Final temperature after annealing")
    parser.add_argument("--min_score_ratio", type=float, default=0.7,
                        help="Min score as fraction of EMA to store in buffer")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--pretrain_model", type=str, default=None,
                        help="Path to pre-trained weights (used with --fresh)")
    args = parser.parse_args()

    if args.mode == "train":
        trainer = AlphaZeroTrainer(
            n_ep=args.n_ep,
            n_mcts=args.n_mcts,
            c=args.c,
            n_chance=args.n_chance,
            lr=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            n_games_per_ep=args.n_games_per_ep,
            temp_anneal_ep=args.temp_anneal_ep,
            temp_final=args.temp_final,
            min_score_ratio=args.min_score_ratio,
            fresh=args.fresh,
            pretrain_model=args.pretrain_model,
        )
        trainer.train()

    elif args.mode == "eval":
        from framework.interaction import InteractionModule
        from framework.logger import RunLogger

        agent = AlphaZeroAgent(
            model_path=args.model,
            n_mcts=args.n_mcts,
            c=args.c,
            n_chance=args.n_chance,
        )
        logger = RunLogger()

        module = InteractionModule(
            config={"grid_size": 4},
            agent=agent,
            logger=logger,
            verbose=True,
            print_board=False,
            num_workers=1,
        )
        module.run(num_games=args.num_games)
        module.print_results()
