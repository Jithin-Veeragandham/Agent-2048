"""
alphazero2048.py
================

AlphaZero-style agent for 2048.

Combines a dual-headed neural network (policy + value) with Monte Carlo
Tree Search (MCTS) guided entirely by the network — no random rollouts.

Key differences from the original AlphaZero (designed for two-player games):
    - 2048 is single-player with stochastic tile spawns.
    - MCTS treats tile spawns as **chance nodes** (expectation over
      possible spawn positions and values) rather than adversary moves.
    - The value head predicts a normalized game score in [0, 1] rather
      than win/loss in {-1, +1}.
    - The policy target is the MCTS visit-count distribution over the
      4 possible moves.

Training loop:
    1. Self-play: use MCTS (guided by current network) to play full games,
       recording (state, π_mcts, final_score) at each step.
    2. Training: sample mini-batches from a replay buffer and minimize
       cross-entropy on policy + MSE on value.
    3. Repeat — the improved network makes MCTS stronger, which generates
       better training data, which further improves the network.

Usage::

    # Train
    python alphazero2048.py --mode train --epochs 200 --sims 100

    # Evaluate
    python alphazero2048.py --mode eval --model alphazero_model.pt --eval-games 100

    # Full pipeline
    python alphazero2048.py --mode full --epochs 200 --sims 100 --eval-games 100
"""

import os
import sys
import time
import json
import math
import copy
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game.engine import Game2048, Action
from agents.base import BaseAgent
from framework.evaluation import RewardFunction

# Minimal heuristic for AlphaZero — tile progress + small empty bonus.
# Heavier shaping (mono, corner) is left for the network to learn.
_AZ_REWARD_FN = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.1, 'mono': 0.0,
    'corner': 0.0, 'merge': 0.0, 'smooth': 0.0,
})
_AZ_HEURISTIC_NORM = 1.1   # approximate upper bound of _AZ_REWARD_FN.compute()


def _heuristic_value(board: np.ndarray) -> float:
    """Board quality in [0, 1] via the AZ reward function."""
    h = _AZ_REWARD_FN.compute(board)
    return float(np.clip(h / _AZ_HEURISTIC_NORM, 0.0, 1.0))

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

NUM_ACTIONS = 4
_VIRTUAL_LOSS = 3   # Penalty applied along selected path so parallel
                    # simulations in the same wave explore different branches.
ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
NUM_TILE_CHANNELS = 16  # exponents 0..15 → tiles 0, 2, 4, ..., 32768


# ═══════════════════════════════════════════════════════════════════
#  INPUT ENCODING — one-hot exponent channels (same as PPO)
# ═══════════════════════════════════════════════════════════════════

def encode_board(x: torch.Tensor) -> torch.Tensor:
    """Convert (N, grid, grid) int board to (N, 16, grid, grid) one-hot."""
    if x.dtype != torch.int64:
        x = x.long()
    exps = torch.zeros_like(x)
    nonzero = x > 0
    exps[nonzero] = torch.log2(x[nonzero].float()).long()
    exps = exps.clamp(0, NUM_TILE_CHANNELS - 1)
    one_hot = F.one_hot(exps, NUM_TILE_CHANNELS).float()
    return one_hot.permute(0, 3, 1, 2)


def encode_board_np(board: np.ndarray) -> torch.Tensor:
    """Encode a single numpy board to a (1, 16, grid, grid) tensor."""
    t = torch.from_numpy(board.astype(np.int64)).unsqueeze(0)
    return encode_board(t)


# ═══════════════════════════════════════════════════════════════════
#  DUAL-HEADED NETWORK — shared conv backbone, policy + value heads
# ═══════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual block with two conv layers and batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNetwork(nn.Module):
    """Dual-headed network: shared backbone → policy head + value head.

    Architecture:
        - Input: one-hot encoded board (16 channels for tile exponents)
        - Backbone: initial conv + N residual blocks
        - Policy head: conv(1x1) → flatten → linear → softmax over 4 actions
        - Value head: conv(1x1) → flatten → linear → tanh (scaled to [0, 1])

    Args:
        grid_size: Board dimension (default 4).
        num_res_blocks: Number of residual blocks in the backbone.
        channels: Number of channels in the backbone convolutions.
    """

    def __init__(self, grid_size: int = 4, num_res_blocks: int = 5,
                 channels: int = 128):
        super().__init__()
        self.grid_size = grid_size

        # ── Shared backbone ──
        self.input_conv = nn.Conv2d(NUM_TILE_CHANNELS, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        flat_size = channels * grid_size * grid_size

        # ── Policy head ──
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * grid_size * grid_size, NUM_ACTIONS)

        # ── Value head ──
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(grid_size * grid_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor,
                legal_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value estimate.

        Args:
            x: Board tensor of shape (N, grid, grid) with int tile values.
            legal_mask: Optional (N, 4) float tensor. 1 = legal, 0 = illegal.

        Returns:
            policy_logits: (N, 4) log-probabilities over actions.
            value: (N,) scalar value estimates in [0, 1].
        """
        # Encode
        x = encode_board(x)

        # Backbone
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        p = self.policy_fc(p)

        # Mask illegal actions before softmax
        if legal_mask is not None:
            p = torch.where(
                legal_mask == 0.0,
                torch.full_like(p, -float('inf')),
                p,
            )

        policy_logits = F.log_softmax(p, dim=-1)

        # Value head — sigmoid maps to [0, 1] for normalized score
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.sigmoid(self.value_fc2(v)).squeeze(-1)

        return policy_logits, v

    def predict(self, board: np.ndarray, device: str = 'cpu'
                ) -> Tuple[np.ndarray, float]:
        """Single-board inference for MCTS.

        Args:
            board: (grid, grid) numpy array.
            device: Torch device string.

        Returns:
            policy: (4,) numpy array of move probabilities.
            value: Scalar value estimate in [0, 1].
        """
        was_training = self.training
        self.eval()
        board_t = torch.from_numpy(board.astype(np.int64)).unsqueeze(0).to(device)

        # Build legal mask
        game_tmp = Game2048.from_state(board)
        available = game_tmp.get_available_moves()
        legal = torch.zeros(1, NUM_ACTIONS, device=device)
        for a in available:
            legal[0, a.value] = 1.0
        if legal.sum() == 0:
            legal[:] = 1.0

        with torch.no_grad():
            log_probs, value = self(board_t, legal_mask=legal)

        if was_training:
            self.train()

        policy = torch.exp(log_probs).cpu().numpy()[0]
        return policy, value.item()


# ═══════════════════════════════════════════════════════════════════
#  MCTS — Neural-network-guided, no random rollouts
# ═══════════════════════════════════════════════════════════════════

class MCTSNode:
    """A node in the MCTS search tree.

    Each node represents a game state. Edges to children represent
    actions (UP/DOWN/LEFT/RIGHT). For 2048 specifically, after an
    action is taken, a random tile spawns — we handle this by
    averaging over a small sample of spawn outcomes.

    Attributes:
        state: The board state at this node.
        parent: Parent node (None for root).
        action: The action that led to this node from parent.
        prior: Prior probability from the neural network policy.
        visit_count: Number of times this node has been visited (N).
        value_sum: Total accumulated value from backpropagation (W).
        children: Dict mapping Action → MCTSNode.
        is_expanded: Whether this node's children have been created.
        is_terminal: Whether the game is over at this state.
    """

    __slots__ = ('state', 'score', 'parent', 'action', 'prior',
                 'visit_count', 'value_sum', 'children',
                 'is_expanded', 'is_terminal')

    def __init__(self, state: np.ndarray, score: int = 0,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[Action] = None,
                 prior: float = 0.0):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Action, 'MCTSNode'] = {}
        self.is_expanded = False
        self.is_terminal = False

    @property
    def q_value(self) -> float:
        """Mean action value Q = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound for Trees (PUCT formula).

        UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N_child)

        This balances exploitation (Q) with exploration (prior × parent visits).
        """
        if self.parent is None:
            return 0.0
        exploration = (c_puct * self.prior
                       * math.sqrt(self.parent.visit_count)
                       / (1 + self.visit_count))
        return self.q_value + exploration


class MCTS:
    """Monte Carlo Tree Search guided by a neural network.

    Instead of random rollouts (as in vanilla MCTS), leaf nodes
    are evaluated by the value head of the network. The policy head
    provides prior probabilities for action selection.

    For 2048's stochasticity: when expanding a node, we simulate
    the action, then **sample a few random tile spawns** and create
    child nodes for each. During selection, we average over these
    spawn outcomes. This is cheaper than enumerating all possible
    spawns (up to 2×empty_cells possibilities).

    Args:
        network: The AlphaZeroNetwork for policy + value inference.
        num_simulations: MCTS iterations per move decision.
        c_puct: Exploration constant in the PUCT formula.
        device: Torch device for network inference.
        num_spawn_samples: How many random tile spawns to sample
            when expanding a node (approximates the expectation).
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_frac: Fraction of prior replaced by Dirichlet noise.
    """

    def __init__(self, network: AlphaZeroNetwork,
                 num_simulations: int = 100,
                 c_puct: float = 1.5,
                 device: str = 'cpu',
                 num_spawn_samples: int = 3,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_frac: float = 0.25,
                 heuristic_weight: float = 0.5):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.num_spawn_samples = num_spawn_samples
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        # Blend: leaf_value = (1-w)*network_value + w*heuristic_value.
        # Starts at 0.5 (equal weight); can be set to 0.0 to disable.
        self.heuristic_weight = heuristic_weight

    def search(self, game: Game2048, temperature: float = 1.0,
               wave_size: int = 8) -> Tuple[np.ndarray, float]:
        """Run MCTS using wave-based batched leaf evaluation.

        Instead of one network call per simulation, each wave selects
        `wave_size` leaves simultaneously (using virtual loss so they
        diverge), evaluates them all in a single batched GPU forward
        pass, then backpropagates. This keeps the GPU busy and cuts
        wall-clock time by ~wave_size× vs. single-leaf evaluation.

        Args:
            game: Current game instance (will be cloned internally).
            temperature: τ for the final move distribution.
            wave_size: Leaves evaluated per network call. Larger =
                better GPU utilisation; too large = lower search quality.
        """
        root_state = game.get_state()
        root = MCTSNode(state=root_state, score=game.get_score())

        # Expand root individually so children exist for Dirichlet noise
        self._expand_single(root)
        self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations:
            # ── Wave: select up to wave_size leaves ──
            paths: List[List[MCTSNode]] = []   # path from root → leaf
            leaves: List[MCTSNode] = []

            for _ in range(wave_size):
                if sims_done + len(leaves) >= self.num_simulations:
                    break
                path = []
                node = root
                while node.is_expanded and not node.is_terminal:
                    node = self._select_child(node)
                    path.append(node)
                # Apply virtual loss so the next selection in this wave
                # is discouraged from picking the same path
                for n in path:
                    n.visit_count += _VIRTUAL_LOSS
                    n.value_sum  -= _VIRTUAL_LOSS
                paths.append(path)
                leaves.append(node)

            if not leaves:
                break

            # ── Batch evaluate all leaves ──
            # Separate terminals (cheap) from non-terminals (need network)
            terminal_mask = [n.is_terminal for n in leaves]
            non_terminal_nodes = [n for n, t in zip(leaves, terminal_mask) if not t]

            # Single batched forward pass for all non-terminal leaves
            if non_terminal_nodes:
                states_np = np.stack([n.state for n in non_terminal_nodes])
                policies_batch, values_batch = self._batch_predict(states_np)
            else:
                policies_batch, values_batch = [], []

            # ── Expand leaves and collect values ──
            nt_idx = 0
            values: List[float] = []
            for node, is_terminal in zip(leaves, terminal_mask):
                if is_terminal:
                    values.append(self._normalize_score(node.score))
                else:
                    policy = policies_batch[nt_idx]
                    value  = float(values_batch[nt_idx])
                    nt_idx += 1
                    if self.heuristic_weight > 0.0:
                        h = _heuristic_value(node.state)
                        value = ((1.0 - self.heuristic_weight) * value
                                 + self.heuristic_weight * h)
                    self._expand_with_policy(node, policy)
                    values.append(value)

            # ── Remove virtual loss and backpropagate real value ──
            for path, node, value in zip(paths, leaves, values):
                for n in path:
                    n.visit_count -= _VIRTUAL_LOSS
                    n.value_sum  += _VIRTUAL_LOSS
                self._backpropagate(node, value)
                sims_done += 1

        policy = self._get_policy(root, temperature)
        return policy, root.q_value

    def _batch_predict(self, states_np: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of boards in one GPU forward pass.

        Returns:
            policies: (B, 4) numpy array of move probabilities.
            values:   (B,)  numpy array of value estimates.
        """
        was_training = self.network.training
        self.network.eval()

        board_t = torch.from_numpy(states_np.astype(np.int64)).to(self.device)

        # Build legal masks for the whole batch
        B = len(states_np)
        legal = torch.zeros(B, NUM_ACTIONS, device=self.device)
        for i, board in enumerate(states_np):
            g = Game2048.from_state(board)
            for a in g.get_available_moves():
                legal[i, a.value] = 1.0
            if legal[i].sum() == 0:
                legal[i] = 1.0

        with torch.no_grad():
            log_probs, values = self.network(board_t, legal_mask=legal)

        if was_training:
            self.network.train()

        policies = torch.exp(log_probs).cpu().numpy()   # (B, 4)
        values   = values.cpu().numpy()                  # (B,)
        return policies, values

    def _expand_single(self, node: MCTSNode) -> float:
        """Expand one leaf with a single network call (used for root only)."""
        game = Game2048.from_state(node.state, score=node.score)
        if game.is_game_over():
            node.is_terminal = True
            node.is_expanded = True
            return self._normalize_score(node.score)

        policies, values = self._batch_predict(node.state[np.newaxis])
        policy = policies[0]
        value  = float(values[0])

        if self.heuristic_weight > 0.0:
            h = _heuristic_value(node.state)
            value = (1.0 - self.heuristic_weight) * value + self.heuristic_weight * h

        available = game.get_available_moves()
        if not available:
            node.is_terminal = True
            node.is_expanded = True
            return self._normalize_score(node.score)

        self._expand_with_policy(node, policy)
        return value

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray):
        """Create children for a node given a precomputed policy array."""
        game = Game2048.from_state(node.state, score=node.score)
        available = game.get_available_moves()
        if not available:
            node.is_terminal = True
            node.is_expanded = True
            return
        for action in available:
            sim = game.clone()
            valid, _ = sim.move(action)
            if valid:
                child = MCTSNode(
                    state=sim.get_state(),
                    score=sim.get_score(),
                    parent=node,
                    action=action,
                    prior=policy[action.value],
                )
                if sim.is_game_over():
                    child.is_terminal = True
                node.children[action] = child
        node.is_expanded = True

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate the value estimate back up to the root."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root priors for exploration.

        This ensures the agent doesn't always follow the network's
        initial policy and can discover better strategies.
        """
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(actions)
        )
        frac = self.dirichlet_frac
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - frac) * child.prior + frac * noise[i]

    def _get_policy(self, root: MCTSNode, temperature: float
                    ) -> np.ndarray:
        """Convert root visit counts to a move probability distribution.

        Args:
            root: The root node after search.
            temperature: τ for the softmax. τ=1 is proportional to
                visit counts; τ→0 is greedy.

        Returns:
            (4,) numpy array of move probabilities.
        """
        visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action, child in root.children.items():
            visits[action.value] = child.visit_count

        if temperature < 1e-3:
            # Greedy: pick the most-visited action
            policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
            if visits.sum() > 0:
                policy[np.argmax(visits)] = 1.0
            else:
                policy[:] = 0.25
            return policy

        # Softmax with temperature
        visits_temp = visits ** (1.0 / temperature)
        total = visits_temp.sum()
        if total > 0:
            return visits_temp / total
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    @staticmethod
    def _normalize_score(score: int) -> float:
        """Normalize a game score to [0, 1] for the value target.

        Uses a logarithmic mapping: log(1 + score) / log(1 + max_expected).
        A score of ~100,000 (very strong 2048 game) maps to ~1.0.
        """
        max_expected = 100000.0
        return math.log(1 + score) / math.log(1 + max_expected)


# ═══════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size buffer storing self-play experience.

    Each entry is a tuple: (board_state, mcts_policy, value_target).
    The value target is the normalized final score of the game
    from which this state was sampled.

    Args:
        capacity: Maximum number of entries. Oldest entries are
            discarded when full (FIFO).
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: np.ndarray,
             value: float):
        """Add a single training example."""
        self.buffer.append((
            state.copy(),
            policy.copy(),
            value,
        ))

    def push_game(self, states: List[np.ndarray],
                  policies: List[np.ndarray],
                  final_score: int,
                  heuristic_weight: float = 0.3):
        """Add all transitions from a completed game.

        Value target per state blends the normalized final score with
        a per-state board heuristic, giving a richer training signal
        (especially early in training when the final score is near 0).

        Args:
            heuristic_weight: Weight for the per-state heuristic
                (0.0 = pure score, 1.0 = pure heuristic).
        """
        z = MCTS._normalize_score(final_score)
        for state, policy in zip(states, policies):
            if heuristic_weight > 0.0:
                h = _heuristic_value(state)
                value = (1.0 - heuristic_weight) * z + heuristic_weight * h
            else:
                value = z
            self.push(state, policy, value)

    def sample(self, batch_size: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch.

        Returns:
            states: (B, grid, grid) int array.
            policies: (B, 4) float array.
            values: (B,) float array.
        """
        indices = np.random.choice(
            len(self.buffer), size=min(batch_size, len(self.buffer)),
            replace=False
        )
        batch = [self.buffer[i] for i in indices]

        states = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)

        return states, policies, values

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════
#  SELF-PLAY — generate training data
# ═══════════════════════════════════════════════════════════════════

def self_play_game(
    network: AlphaZeroNetwork,
    config: Dict,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    device: str = 'cpu',
    temperature_threshold: int = 30,
    heuristic_weight: float = 0.5,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Play one complete game using MCTS, collecting training data.

    For the first `temperature_threshold` moves, use temperature=1
    for exploration. After that, switch to temperature≈0 (greedy)
    for stronger play.

    Args:
        network: Current AlphaZeroNetwork.
        config: Game config dict.
        num_simulations: MCTS iterations per move.
        c_puct: Exploration constant for MCTS.
        device: Torch device.
        temperature_threshold: Move number after which τ → 0.

    Returns:
        states: List of board states visited during the game.
        policies: List of MCTS-derived policy distributions.
        final_score: The game's final merge score.
    """
    game = Game2048(config)
    mcts = MCTS(network, num_simulations, c_puct, device,
                heuristic_weight=heuristic_weight)

    states = []
    policies = []
    move_count = 0

    while not game.is_game_over():
        available = game.get_available_moves()
        if not available:
            break

        # Temperature scheduling
        temperature = 1.0 if move_count < temperature_threshold else 0.1

        # Record current state
        state = game.get_state()
        states.append(state)

        # Run MCTS search
        policy, _ = mcts.search(game, temperature=temperature)
        policies.append(policy)

        # Select action from the MCTS policy
        if temperature < 0.5:
            # Near-greedy
            action_idx = np.argmax(policy)
        else:
            # Sample from the distribution
            action_idx = np.random.choice(NUM_ACTIONS, p=policy)

        action = ACTIONS[action_idx]

        # Ensure we pick a legal action
        if action not in available:
            action = available[0]

        game.move(action)
        move_count += 1

    return states, policies, game.get_score()


# ═══════════════════════════════════════════════════════════════════
#  ALPHAZERO TRAINER
# ═══════════════════════════════════════════════════════════════════

class AlphaZeroTrainer:
    """Full AlphaZero training pipeline: self-play → train → repeat.

    Each epoch consists of:
        1. Self-play: play `games_per_epoch` games using MCTS
           guided by the current network, storing (state, π, z)
           tuples in a replay buffer.
        2. Training: sample mini-batches from the buffer and
           update the network to minimize:
               L = CE(π_mcts, π_net) + MSE(z, v_net) + c·‖θ‖²

    Args:
        config: Game configuration dict.
        grid_size: Board dimension.
        num_res_blocks: Residual blocks in the network.
        channels: Conv channels in the network.
        num_simulations: MCTS simulations per move.
        c_puct: MCTS exploration constant.
        lr: Learning rate.
        weight_decay: L2 regularization.
        buffer_capacity: Replay buffer size.
        batch_size: Mini-batch size for training.
        epochs: Number of self-play + training cycles.
        games_per_epoch: Self-play games per epoch.
        train_steps_per_epoch: Gradient updates per epoch.
        device: Torch device (auto-detected if None).
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        grid_size: int = 4,
        num_res_blocks: int = 5,
        channels: int = 128,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        epochs: int = 200,
        games_per_epoch: int = 25,
        train_steps_per_epoch: int = 100,
        device: Optional[str] = None,
        heuristic_weight: float = 0.5,
    ):
        self.config = config or {'grid_size': grid_size}
        self.grid_size = self.config.get('grid_size', grid_size)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.epochs = epochs
        self.games_per_epoch = games_per_epoch
        self.train_steps_per_epoch = train_steps_per_epoch
        self.heuristic_weight = heuristic_weight

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.network = AlphaZeroNetwork(
            grid_size=self.grid_size,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.training_history: List[Dict] = []
        self.total_games: int = 0
        self.historical_max_tile: int = 0
        self.total_training_time: float = 0.0

    def train(self, save_path: Optional[str] = None,
              log_interval: int = 1):
        """Run the full AlphaZero training loop.

        Args:
            save_path: Where to save the final model.
            log_interval: Print stats every N epochs.
        """
        print(f"\n{'=' * 65}", flush=True)
        print(f"  AlphaZero Training", flush=True)
        print(f"  {self.grid_size}×{self.grid_size} | "
              f"{self.num_simulations} MCTS sims/move | "
              f"{self.epochs} epochs", flush=True)
        print(f"  {self.games_per_epoch} games/epoch | "
              f"{self.train_steps_per_epoch} train steps/epoch", flush=True)
        print(f"  LR={self.optimizer.param_groups[0]['lr']:.1e} | "
              f"Batch={self.batch_size} | Device: {self.device}", flush=True)
        print(f"  Network: {sum(p.numel() for p in self.network.parameters()):,} params", flush=True)
        print(f"{'=' * 65}\n", flush=True)

        os.makedirs("models", exist_ok=True)
        start_time = time.time()
        checkpoint_interval = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # ── Phase 1: Self-play ──
            self.network.eval()
            sp_states_total = 0
            sp_scores = []
            sp_max_tiles = []

            for g in range(self.games_per_epoch):
                print(
                    f"  Epoch {epoch}/{self.epochs} — self-play "
                    f"{g + 1}/{self.games_per_epoch} ...",
                    end="\r", flush=True,
                )
                states, policies, final_score = self_play_game(
                    network=self.network,
                    config=self.config,
                    num_simulations=self.num_simulations,
                    c_puct=self.c_puct,
                    device=self.device,
                    heuristic_weight=self.heuristic_weight,
                )

                # Push entire game to replay buffer
                self.buffer.push_game(states, policies, final_score,
                                      heuristic_weight=self.heuristic_weight)
                sp_states_total += len(states)
                sp_scores.append(final_score)

                # Track max tile
                if states:
                    max_tile = int(np.max(states[-1]))
                    sp_max_tiles.append(max_tile)
                    self.historical_max_tile = max(
                        self.historical_max_tile, max_tile
                    )

                self.total_games += 1

            sp_time = time.time() - epoch_start

            # ── Phase 2: Network training ──
            train_start = time.time()
            self.network.train()

            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_updates = 0

            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps_per_epoch):
                    states_np, policies_np, values_np = self.buffer.sample(
                        self.batch_size
                    )

                    states_t = torch.from_numpy(states_np).to(
                        dtype=torch.int64, device=self.device
                    )
                    target_policies = torch.from_numpy(policies_np).to(
                        dtype=torch.float32, device=self.device
                    )
                    target_values = torch.from_numpy(values_np).to(
                        dtype=torch.float32, device=self.device
                    )

                    # Build legal mask from MCTS policy targets:
                    # actions with non-zero target probability were legal
                    legal_mask = (target_policies > 0).float()

                    # Forward pass
                    pred_log_policy, pred_value = self.network(
                        states_t, legal_mask=legal_mask
                    )

                    # Policy loss: cross-entropy with MCTS targets
                    # = -Σ π_mcts(a) · log π_net(a)
                    policy_loss = -(target_policies * pred_log_policy).sum(dim=1).mean()

                    # Value loss: MSE between predicted and actual game value
                    value_loss = F.mse_loss(pred_value, target_values)

                    # Combined loss
                    loss = policy_loss + value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_updates += 1

            self.scheduler.step()
            train_time = time.time() - train_start
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time

            # ── Logging ──
            avg_pl = total_policy_loss / max(num_updates, 1)
            avg_vl = total_value_loss / max(num_updates, 1)
            avg_score = np.mean(sp_scores) if sp_scores else 0
            avg_max_tile = int(np.mean(sp_max_tiles)) if sp_max_tiles else 0

            self.training_history.append({
                'epoch': epoch,
                'avg_score': round(float(avg_score), 1),
                'max_tile_this_epoch': max(sp_max_tiles) if sp_max_tiles else 0,
                'historical_max_tile': self.historical_max_tile,
                'policy_loss': round(avg_pl, 4),
                'value_loss': round(avg_vl, 4),
                'buffer_size': len(self.buffer),
                'sp_states': sp_states_total,
                'sp_time_sec': round(sp_time, 1),
                'train_time_sec': round(train_time, 1),
                'total_games': self.total_games,
            })

            if epoch % log_interval == 0:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:>4}/{self.epochs}  |  "
                    f"Score: {avg_score:>8.0f}  |  "
                    f"Max tile: {self.historical_max_tile:>5}  |  "
                    f"PL: {avg_pl:.4f}  VL: {avg_vl:.4f}  |  "
                    f"Buf: {len(self.buffer):>6}  |  "
                    f"SP: {sp_time:.1f}s  Train: {train_time:.1f}s  |  "
                    f"LR: {lr:.1e}",
                    flush=True,
                )

            # ── Checkpoint ──
            if (epoch + 1) % checkpoint_interval == 0 and epoch > 0:
                ckpt_path = f"models/alphazero_ep{epoch + 1}.pt"
                self.save_model(ckpt_path)

        total_time = time.time() - start_time
        print(f"\n{'=' * 65}", flush=True)
        print(f"  Done in {total_time:.1f}s | "
              f"Games: {self.total_games:,} | "
              f"Max tile: {self.historical_max_tile}", flush=True)
        print(f"  Buffer: {len(self.buffer):,} entries", flush=True)
        print(f"{'=' * 65}\n", flush=True)

        if save_path:
            self.save_model(save_path)

    def save_model(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grid_size': self.grid_size,
            'total_games': self.total_games,
            'training_history': self.training_history,
            'historical_max_tile': self.historical_max_tile,
            'total_training_time': self.total_training_time,
        }, path)
        print(f"  Saved: {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.total_games = ckpt.get('total_games', 0)
        self.training_history = ckpt.get('training_history', [])
        self.historical_max_tile = ckpt.get('historical_max_tile', 0)
        self.total_training_time = ckpt.get('total_training_time', 0.0)
        print(f"  Loaded: {path}")


# ═══════════════════════════════════════════════════════════════════
#  ALPHAZERO AGENT — BaseAgent wrapper for InteractionModule
# ═══════════════════════════════════════════════════════════════════

class AlphaZeroAgent(BaseAgent):
    """Trained AlphaZero agent for evaluation via InteractionModule.

    At inference time, runs MCTS with the trained network to select
    moves. The number of MCTS simulations can be adjusted to trade
    off compute vs performance.

    Args:
        model_path: Path to a saved AlphaZeroNetwork checkpoint.
        grid_size: Board dimension.
        num_simulations: MCTS iterations per move at evaluation time.
            More sims = stronger play but slower inference.
        c_puct: Exploration constant for MCTS.
        deterministic: If True, always pick the most-visited action
            (temperature → 0). If False, sample proportionally.
        device: Torch device.
    """

    def __init__(self, model_path: Optional[str] = None,
                 grid_size: int = 4,
                 num_simulations: int = 100,
                 c_puct: float = 1.5,
                 deterministic: bool = True,
                 device: Optional[str] = None):
        super().__init__("AlphaZero")
        self.agent_type = "alphazero"
        self.grid_size = grid_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.deterministic = deterministic

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.network = AlphaZeroNetwork(grid_size=grid_size).to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.network.load_state_dict(ckpt['model_state_dict'])
            else:
                self.network.load_state_dict(ckpt)
            print(f"AlphaZero: loaded {model_path}")
        else:
            print("AlphaZero: randomly initialized (untrained)")

        self.network.eval()
        self.mcts = MCTS(
            self.network, num_simulations, c_puct, self.device,
        )

    def choose_action(self, state: np.ndarray,
                      available_moves: List[Action],
                      game_context: Optional[Dict] = None) -> Action:
        """Select an action using MCTS search.

        Args:
            state: Current board state (grid × grid numpy array).
            available_moves: List of legal actions.
            game_context: Dict from InteractionModule containing
                'game' (for cloning) and other context.

        Returns:
            The chosen Action.
        """
        # Reconstruct a game for MCTS to clone internally
        if game_context and 'game' in game_context:
            game = game_context['game']
        else:
            score = game_context.get('score', 0) if game_context else 0
            game = Game2048.from_state(state, score=score)

        # Temperature: greedy at eval time
        temperature = 0.01 if self.deterministic else 1.0

        # Run MCTS
        policy, _ = self.mcts.search(game, temperature=temperature)

        # Select action
        if self.deterministic:
            # Mask illegal actions and pick best
            masked = policy.copy()
            legal_indices = {a.value for a in available_moves}
            for i in range(NUM_ACTIONS):
                if i not in legal_indices:
                    masked[i] = -1.0
            action_idx = np.argmax(masked)
        else:
            # Mask and renormalize
            masked = policy.copy()
            legal_indices = {a.value for a in available_moves}
            for i in range(NUM_ACTIONS):
                if i not in legal_indices:
                    masked[i] = 0.0
            total = masked.sum()
            if total > 0:
                masked /= total
            else:
                masked = np.ones(NUM_ACTIONS) / NUM_ACTIONS
            action_idx = np.random.choice(NUM_ACTIONS, p=masked)

        chosen = ACTIONS[action_idx]
        if chosen not in available_moves:
            chosen = available_moves[0]
        return chosen

    def get_params(self) -> Dict:
        """Return agent parameters for logging."""
        return {
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'deterministic': self.deterministic,
        }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero 2048")
    parser.add_argument("--mode", choices=["train", "eval", "full"],
                        default="full")
    parser.add_argument("--model", default="alphazero_model.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--games-per-epoch", type=int, default=25)
    parser.add_argument("--train-steps", type=int, default=100,
                        help="Gradient updates per epoch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--res-blocks", type=int, default=5)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=100,
                        help="MCTS simulations per move during eval")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded {args.config} "
              f"(grid: {config.get('grid_size', 4)}×"
              f"{config.get('grid_size', 4)})")
    except FileNotFoundError:
        config = {"grid_size": 4}
        print(f"{args.config} not found, using defaults (4×4)")

    grid_size = config.get("grid_size", 4)

    if args.mode == "train":
        trainer = AlphaZeroTrainer(
            config=config,
            num_res_blocks=args.res_blocks,
            channels=args.channels,
            num_simulations=args.sims,
            c_puct=args.c_puct,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            games_per_epoch=args.games_per_epoch,
            train_steps_per_epoch=args.train_steps,
        )
        if args.resume:
            trainer.load_model(args.resume)
        trainer.train(save_path=args.model)

    elif args.mode == "eval":
        from framework.interaction import InteractionModule
        from framework.logger import RunLogger
        logger = RunLogger()
        agent = AlphaZeroAgent(
            model_path=args.model,
            grid_size=grid_size,
            num_simulations=args.eval_sims,
            deterministic=True,
        )
        module = InteractionModule(
            config=config, agent=agent, logger=logger, verbose=True,
        )
        module.run(num_games=args.eval_games)
        module.print_results()

    elif args.mode == "full":
        trainer = AlphaZeroTrainer(
            config=config,
            num_res_blocks=args.res_blocks,
            channels=args.channels,
            num_simulations=args.sims,
            c_puct=args.c_puct,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            games_per_epoch=args.games_per_epoch,
            train_steps_per_epoch=args.train_steps,
        )
        if args.resume:
            trainer.load_model(args.resume)
        trainer.train(save_path=args.model)

        from framework.interaction import InteractionModule
        from framework.logger import RunLogger
        logger = RunLogger()
        agent = AlphaZeroAgent(
            model_path=args.model,
            grid_size=grid_size,
            num_simulations=args.eval_sims,
            deterministic=True,
        )
        module = InteractionModule(
            config=config, agent=agent, logger=logger, verbose=True,
        )
        module.set_training_stats(
            training_time_sec=trainer.total_training_time,
            training_episodes=trainer.total_games,
        )
        module.run(num_games=args.eval_games)
        module.print_results()