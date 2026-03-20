import time
import json
import os
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Enable CuDNN autotuner and TF32
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from game import Game2048, Action
from interaction import BaseAgent
from evaluation import RewardFunction

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

NUM_ACTIONS = 4
ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


# ═══════════════════════════════════════════════════════════════════
#  VECTORIZED ENV — wraps game.py's Game2048 for batch operations
#  Returns torch tensors on the specified device for GPU training.
# ═══════════════════════════════════════════════════════════════════

class GameVectorizedEnv:
    """Runs N 2048 games in parallel using game.py's Game2048.

    Each environment is a real Game2048 instance. Board states are
    synced to a shared torch tensor for GPU-based neural network
    inference. All game logic flows through game.py.
    """

    def __init__(self, num_envs: int, config: Dict, device: str = 'cpu'):
        self.num_envs = num_envs
        self.config = config
        self.grid_size = config.get('grid_size', 4)
        self.device = device
        self.games = [Game2048(config) for _ in range(num_envs)]
        self.state = self._sync_state()
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.score = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def _sync_state(self) -> torch.Tensor:
        """Read all Game2048 boards into a single (N, grid, grid) tensor."""
        boards = np.stack([g.get_state() for g in self.games])
        return torch.from_numpy(boards).to(dtype=torch.int64, device=self.device)

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset all environments."""
        self.games = [Game2048(self.config) for _ in range(self.num_envs)]
        self.state = self._sync_state()
        self.done.zero_()
        self.score.zero_()
        return self.state.clone(), self.done.clone()

    def reset_envs(self, mask: torch.Tensor):
        """Reset only environments where mask is True."""
        if not mask.any():
            return
        mask_np = mask.cpu().numpy()
        for i in range(self.num_envs):
            if mask_np[i]:
                self.games[i] = Game2048(self.config)
                self.score[i] = 0.0
        self.state = self._sync_state()
        self.done[mask] = False

    def step(self, actions: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all envs using game.py. Returns (next_state, rewards, dones)."""
        _, next_state, rewards, dones = self.step_with_legal(actions)
        return next_state, rewards, dones

    def step_with_legal(self, actions: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all envs AND compute legal mask for next state.

        Returns (legal_mask_for_next_state, next_state, rewards, dones).
        """
        actions_np = actions.cpu().numpy()
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i, g in enumerate(self.games):
            action = ACTIONS[int(actions_np[i])]
            valid, reward = g.move(action)
            rewards[i] = float(reward)
            dones[i] = g.is_game_over()

        self.state = self._sync_state()
        self.done = dones
        self.score += rewards

        # Compute legal mask for the NEW state
        legal = torch.zeros(self.num_envs, 4, device=self.device)
        for i, g in enumerate(self.games):
            available = g.get_available_moves()
            for a in available:
                legal[i, a.value] = 1.0
        # If no legal moves, allow all (game is done anyway)
        no_legal = legal.sum(dim=1) == 0
        legal[no_legal] = 1.0

        return legal, self.state.clone(), rewards.clone(), dones.clone()

    def get_legal_mask(self) -> torch.Tensor:
        """Return (N, 4) float tensor of legal moves using game.py."""
        legal = torch.zeros(self.num_envs, 4, device=self.device)
        for i, g in enumerate(self.games):
            available = g.get_available_moves()
            for a in available:
                legal[i, a.value] = 1.0
        no_legal = legal.sum(dim=1) == 0
        legal[no_legal] = 1.0
        return legal


# ═══════════════════════════════════════════════════════════════════
#  NEURAL NETWORK — same architecture as reference
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  INPUT ENCODING — one-hot exponent channels
# ═══════════════════════════════════════════════════════════════════

NUM_TILE_CHANNELS = 16  # exponents 0..15 → tiles 0, 2, 4, ..., 32768


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


class Critic(nn.Module):
    def __init__(self, grid_size=4, hidden_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_TILE_CHANNELS, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        conv_out = grid_size * grid_size * 64
        self.linear1 = nn.Linear(conv_out, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = encode_board(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.value_head(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, grid_size=4, hidden_size=256, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_TILE_CHANNELS, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        conv_out = grid_size * grid_size * 64
        self.linear1 = nn.Linear(conv_out, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64)
        self.policy_head = nn.Linear(64, num_actions)

    def forward(self, x, legal_actions=None):
        x = encode_board(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.policy_head(x)
        if legal_actions is not None:
            logits = torch.where(
                legal_actions == 0.0,
                torch.full_like(logits, -float('inf')),
                logits,
            )
        return F.log_softmax(logits, dim=-1)


class PPONetworkRef(nn.Module):
    """Combined actor-critic with shared conv backbone."""

    def __init__(self, grid_size=4, hidden_size=256):
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = 4
        self.actor = Actor(grid_size, hidden_size, self.num_actions)
        self.critic = Critic(grid_size, hidden_size)
        # Share conv backbone
        self.critic.conv1 = self.actor.conv1
        self.critic.bn1 = self.actor.bn1
        self.critic.conv2 = self.actor.conv2
        self.critic.bn2 = self.actor.bn2

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, legal_actions=None):
        log_probs = self.actor(x, legal_actions)
        policy_dist = Categorical(logits=log_probs)
        if action is None:
            action = policy_dist.sample()
        value = self.critic(x)
        return action, policy_dist.log_prob(action), policy_dist.entropy(), value


# ═══════════════════════════════════════════════════════════════════
#  GAE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_gae_advantages(
    rewards: torch.Tensor, values: torch.Tensor,
    dones: torch.Tensor, next_val: torch.Tensor,
    gamma: float, gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = next_val
            next_non_terminal = 1.0 - dones[t]
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns



# ── Default weights (matching evaluation.py's RewardFunction.DEFAULT_WEIGHTS)
SHAPING_WEIGHTS = {
        'tile':   1.0,
        'empty':  0.5,
        'mono':   2.5,
        'merge':  0.5,
        'smooth': 0.1,
}


def _tile_score_batch(boards: torch.Tensor) -> torch.Tensor:
    """Batched tile_score: sum of value * log2(value) for non-zero tiles.

    Args:
        boards: (N, grid, grid) int tensor

    Returns:
        (N,) float tensor
    """
    flat = boards.reshape(boards.shape[0], -1).float()         # (N, G*G)
    nonzero_mask = flat > 0
    log_vals = torch.zeros_like(flat)
    log_vals[nonzero_mask] = torch.log2(flat[nonzero_mask])
    # value * log2(value), zeroed where board == 0
    weighted = flat * log_vals                                  # (N, G*G)
    return weighted.sum(dim=1)                                  # (N,)


def _empty_bonus_batch(boards: torch.Tensor) -> torch.Tensor:
    """Batched empty_bonus: empty_count² + adjacent empty-empty pairs.

    Args:
        boards: (N, grid, grid) int tensor

    Returns:
        (N,) float tensor
    """
    empty = (boards == 0)                                       # (N, H, W) bool
    # Count component: squared
    count = empty.reshape(boards.shape[0], -1).sum(dim=1).float()
    count_score = count ** 2                                    # (N,)

    # Adjacency: horizontal empty-empty pairs
    h_adj = (empty[:, :, :-1] & empty[:, :, 1:])               # (N, H, W-1)
    h_count = h_adj.reshape(boards.shape[0], -1).sum(dim=1).float()

    # Adjacency: vertical empty-empty pairs
    v_adj = (empty[:, :-1, :] & empty[:, 1:, :])               # (N, H-1, W)
    v_count = v_adj.reshape(boards.shape[0], -1).sum(dim=1).float()

    return count_score + h_count + v_count                      # (N,)


def _merge_potential_batch(boards: torch.Tensor) -> torch.Tensor:
    """Batched merge_potential: sum of log2(val) for adjacent equal non-zero pairs.

    Args:
        boards: (N, grid, grid) int tensor

    Returns:
        (N,) float tensor
    """
    flat_f = boards.float()
    log_board = torch.zeros_like(flat_f)
    nonzero = flat_f > 0
    log_board[nonzero] = torch.log2(flat_f[nonzero])

    # Horizontal: same value and both non-zero
    h_match = (boards[:, :, :-1] == boards[:, :, 1:]) & (boards[:, :, :-1] > 0)
    # Weight by log2 of the matching value
    h_weight = log_board[:, :, :-1] * h_match.float()           # (N, H, W-1)
    h_score = h_weight.reshape(boards.shape[0], -1).sum(dim=1)

    # Vertical: same value and both non-zero
    v_match = (boards[:, :-1, :] == boards[:, 1:, :]) & (boards[:, :-1, :] > 0)
    v_weight = log_board[:, :-1, :] * v_match.float()           # (N, H-1, W)
    v_score = v_weight.reshape(boards.shape[0], -1).sum(dim=1)

    return h_score + v_score                                    # (N,)


def _smoothness_batch(boards: torch.Tensor) -> torch.Tensor:
    """Batched smoothness: sum of |log2(a) - log2(b)| for adjacent non-zero pairs.

    Args:
        boards: (N, grid, grid) int tensor

    Returns:
        (N,) float tensor — penalty (higher = rougher board)
    """
    flat_f = boards.float()
    log_board = torch.zeros_like(flat_f)
    nonzero = flat_f > 0
    log_board[nonzero] = torch.log2(flat_f[nonzero])

    # Horizontal: both non-zero
    h_both = (boards[:, :, :-1] > 0) & (boards[:, :, 1:] > 0)
    h_diff = torch.abs(log_board[:, :, :-1] - log_board[:, :, 1:]) * h_both.float()
    h_penalty = h_diff.reshape(boards.shape[0], -1).sum(dim=1)

    # Vertical: both non-zero
    v_both = (boards[:, :-1, :] > 0) & (boards[:, 1:, :] > 0)
    v_diff = torch.abs(log_board[:, :-1, :] - log_board[:, 1:, :]) * v_both.float()
    v_penalty = v_diff.reshape(boards.shape[0], -1).sum(dim=1)

    return h_penalty + v_penalty                                # (N,)

def _monotonicity_batch(boards: torch.Tensor) -> torch.Tensor:
    """Batched monotonicity: reward for consistent value gradients per row/col.
 
    For each row, computes how well values decrease left-to-right vs
    right-to-left, picks the better direction, and sums. Same for columns.
    Uses log₂ scale so tile magnitude differences are proportional.
 
    Matches evaluation.py's RewardFunction.monotonicity() but batched on GPU.
 
    Args:
        boards: (N, grid, grid) int tensor
 
    Returns:
        (N,) float tensor — monotonicity score (higher = more monotonic)
    """
    flat_f = boards.float()
    log_board = torch.zeros_like(flat_f)
    nonzero = flat_f > 0
    log_board[nonzero] = torch.log2(flat_f[nonzero])
 
    # ── Horizontal: per-row best direction ──
    h_diff = log_board[:, :, :-1] - log_board[:, :, 1:]        # (N, H, W-1)
    h_both = (boards[:, :, :-1] > 0) & (boards[:, :, 1:] > 0) # both non-zero
    h_diff_masked = h_diff * h_both.float()
 
    # Per-row: sum positive diffs (left > right) and negative diffs (right > left)
    h_left = F.relu(h_diff_masked).sum(dim=2)                  # (N, H)
    h_right = F.relu(-h_diff_masked).sum(dim=2)                # (N, H)
    h_mono = torch.max(h_left, h_right).sum(dim=1)             # (N,)
 
    # ── Vertical: per-column best direction ──
    v_diff = log_board[:, :-1, :] - log_board[:, 1:, :]        # (N, H-1, W)
    v_both = (boards[:, :-1, :] > 0) & (boards[:, 1:, :] > 0)
    v_diff_masked = v_diff * v_both.float()
 
    v_up = F.relu(v_diff_masked).sum(dim=1)                    # (N, W)
    v_down = F.relu(-v_diff_masked).sum(dim=1)                 # (N, W)
    v_mono = torch.max(v_up, v_down).sum(dim=1)                # (N,)
 
    return h_mono + v_mono   

def _heuristic_value(boards: torch.Tensor,
                     weights: dict = SHAPING_WEIGHTS) -> torch.Tensor:
    """Compute composite heuristic value for a batch of boards.

    Matches evaluation.py's RewardFunction.compute() but batched on GPU.
    Returns normalized, weighted composite score per board.

    Args:
        boards: (N, grid, grid) int tensor
        weights: dict with keys 'tile', 'empty', 'merge', 'smooth'

    Returns:
        (N,) float tensor — composite heuristic value
    """
    gs = boards.shape[-1]
    max_cells = gs * gs  # e.g. 16 for 4x4

    t = _tile_score_batch(boards)
    e = _empty_bonus_batch(boards)
    mo = _monotonicity_batch(boards)
    m = _merge_potential_batch(boards)
    s = _smoothness_batch(boards)

    # Normalization constants (matching evaluation.py)
    t_norm = t / (2048.0 * 11.0 * max_cells + 1e-8)
    e_norm = e / (max_cells ** 2 + max_cells * 2.0 + 1e-8)
    mo_norm = mo / (11.0 * max_cells + 1e-8)
    m_norm = m / (11.0 * max_cells + 1e-8)
    s_norm = s / (11.0 * max_cells * 2.0 + 1e-8)

    return (weights['tile'] * t_norm
            + weights['empty'] * e_norm
            + weights['mono'] * mo_norm
            + weights['merge'] * m_norm
            - weights['smooth'] * s_norm)


def compute_shaped_rewards(
    state: torch.Tensor, next_state: torch.Tensor,
    rew: torch.Tensor, num_envs: int,
    gamma: float = 0.998,
    weights: dict = SHAPING_WEIGHTS,
) -> torch.Tensor:
    """Potential-based reward shaping using evaluation.py heuristics.

    shaped_reward = merge_reward + scale * (gamma * h(s') - h(s))

    where h() is the composite heuristic from evaluation.py,
    computed entirely in batched tensor ops.

    Args:
        state:      (N, grid, grid) board before move
        next_state: (N, grid, grid) board after move
        rew:        (N,) merge reward from environment
        num_envs:   number of environments (N)
        gamma:      discount factor for potential shaping
        weights:    heuristic component weights

    Returns:
        (N,) shaped reward tensor
    """
    h_old = _heuristic_value(state, weights)
    h_new = _heuristic_value(next_state, weights)

    # Potential-based shaping: gamma * phi(s') - phi(s)
    # Scale factor controls how much shaping influences vs raw merge reward
    shaping_scale = 10.0
    potential_bonus = shaping_scale * (gamma * h_new - h_old)

    return rew + potential_bonus


# ═══════════════════════════════════════════════════════════════════
#  PPO TRAINER
# ═══════════════════════════════════════════════════════════════════

class PPOTrainer:
    """PPO training with tensor-based env, legal action masking,
    and multi-epoch mini-batch updates."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        grid_size: int = 4,
        num_envs: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.998,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.001,
        vf_coef: float = 0.5,
        rollout_length: int = 64,
        epochs: int = 100000,
        ppo_epochs: int = 3,
        mini_batch_size: int = 1024,
        device: str = None,
    ):
        self.config = config or {'grid_size': grid_size}
        self.grid_size = self.config.get('grid_size', grid_size)
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.total_epochs = epochs
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.ent_coef = ent_coef_start  # current value, annealed during training
        self.vf_coef = vf_coef

        self.network = PPONetworkRef(grid_size=self.grid_size).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.scaler = torch.amp.GradScaler(device=self.device)

        # Use game.py-backed env
        self.env = GameVectorizedEnv(num_envs, self.config, self.device)

        self.training_history: List[Dict] = []
        self.total_timesteps: int = 0
        self.total_episodes: int = 0
        self.historical_max_tile: int = 0

    def train(self, use_shaping: bool = True, log_interval: int = 10,
              save_path: Optional[str] = None):
        grid_size = self.grid_size
        num_envs = self.num_envs
        rollout_length = self.rollout_length
        device = self.device
        epochs = self.total_epochs
        checkpoint_interval = min(5000, max(1, epochs // 10))

        network = self.network
        optimizer = self.optimizer
        scaler = self.scaler
        env = self.env

        os.makedirs("models", exist_ok=True)

        batch_size = num_envs * rollout_length

        print(f"\n{'=' * 65}")
        print(f"  PPO Training (game.py Engine)")
        print(f"  {grid_size}x{grid_size} | {num_envs} envs | "
              f"{rollout_length} steps/rollout | {epochs} epochs")
        print(f"  LR={self.optimizer.param_groups[0]['lr']:.1e} "
              f"gamma={self.gamma} clip={self.clip_eps} "
              f"| Device: {device}")
        print(f"  Steps/update: {batch_size:,} | "
              f"PPO epochs: {self.ppo_epochs} | "
              f"Mini-batch: {self.mini_batch_size}")
        print(f"  Entropy: {self.ent_coef_start} → {self.ent_coef_end} "
              f"(linear anneal)")
        print(f"{'=' * 65}\n")

        state, done = env.reset()
        start_time = time.time()

        # Preallocate rollout buffers ONCE (not every epoch!)
        states_buf = torch.empty(
            rollout_length, num_envs, grid_size, grid_size,
            device=device, dtype=torch.int64
        )
        actions_buf = torch.empty(
            rollout_length, num_envs, dtype=torch.int64, device=device
        )
        rewards_buf = torch.empty(rollout_length, num_envs, device=device)
        dones_buf = torch.empty(rollout_length, num_envs, device=device)
        values_buf = torch.empty(rollout_length, num_envs, device=device)
        logp_buf = torch.empty(rollout_length, num_envs, device=device)
        legal_buf = torch.empty(
            rollout_length, num_envs, 4, device=device
        )

        try:
            from tqdm import tqdm
            ep_iter = tqdm(range(epochs), desc="Training")
        except ImportError:
            ep_iter = range(epochs)

        # Compute initial legal mask (only needed once at start)
        legal_actions = env.get_legal_mask()

        for ep in ep_iter:
            # ── Collect rollout ──
            network.eval()
            for t in range(rollout_length):
                # legal_actions is already computed (from previous step or init)

                # Get action from policy with legal masking
                with torch.no_grad():
                    action, logp, _, val = network.get_action_and_value(
                        state, legal_actions=legal_actions
                    )

                # Store BEFORE stepping (legal_actions is for current state)
                states_buf[t].copy_(state)
                actions_buf[t] = action
                values_buf[t] = val
                logp_buf[t] = logp
                legal_buf[t] = legal_actions

                # Fused step: steps env AND computes legal mask for next state
                next_legal, next_state, rew, done = env.step_with_legal(action)

                # Reward shaping
                if use_shaping:
                    rew = compute_shaped_rewards(
                        state, next_state, rew, num_envs,
                        gamma=self.gamma,
                    )

                rewards_buf[t] = rew
                dones_buf[t] = done.float()

                state = next_state
                legal_actions = next_legal

                # Partial reset: only finished envs
                if done.any():
                    env.reset_envs(done)
                    state = env.state.clone()
                    # Recompute legal mask for reset envs
                    legal_actions = env.get_legal_mask()
                    self.total_episodes += int(done.sum().item())

                # Track max tile
                curr_max = int(next_state.max().item())
                self.historical_max_tile = max(
                    self.historical_max_tile, curr_max
                )

            self.total_timesteps += rollout_length * num_envs

            # ── Compute GAE ──
            with torch.no_grad():
                next_val = network.get_value(state)
            advantages_t, returns_t = compute_gae_advantages(
                rewards_buf, values_buf, dones_buf, next_val,
                self.gamma, self.gae_lambda
            )

            # ── Flatten ──
            states_b = states_buf.reshape(-1, grid_size, grid_size)
            actions_b = actions_buf.reshape(-1)
            adv_flat = advantages_t.reshape(-1)
            advantages_b = (
                (adv_flat - adv_flat.mean())
                / (adv_flat.std(unbiased=False) + 1e-8)
            ).detach()
            returns_b = returns_t.reshape(-1).detach()
            logp_old_b = logp_buf.reshape(-1).detach()
            values_old_b = values_buf.reshape(-1).detach()
            legal_b = legal_buf.reshape(-1, 4).detach()

            # ── PPO multi-epoch mini-batch update ──
            network.train()
            indices = torch.arange(batch_size, device=device)
            mb_size = min(self.mini_batch_size, batch_size)

            total_ploss = 0.0
            total_vloss = 0.0
            total_ent = 0.0
            n_updates = 0

            for _ in range(self.ppo_epochs):
                # Shuffle
                perm = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, mb_size):
                    end = min(start + mb_size, batch_size)
                    mb_idx = perm[start:end]

                    mb_states = states_b[mb_idx]
                    mb_actions = actions_b[mb_idx]
                    mb_returns = returns_b[mb_idx]
                    mb_logp_old = logp_old_b[mb_idx]
                    mb_advantages = advantages_b[mb_idx]
                    mb_values_old = values_old_b[mb_idx]
                    mb_legal = legal_b[mb_idx]

                    optimizer.zero_grad()

                    with torch.amp.autocast(
                        device_type=device if device != 'mps' else 'cpu'
                    ):
                        _, new_logp, entropy, new_val = \
                            network.get_action_and_value(
                                mb_states, mb_actions,
                                legal_actions=mb_legal
                            )

                        ratio = torch.exp(new_logp - mb_logp_old)
                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(
                            ratio,
                            1 - self.clip_eps,
                            1 + self.clip_eps
                        ) * mb_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        vpred = new_val.view(-1)
                        v_unclipped = (vpred - mb_returns).pow(2)
                        v_clipped = mb_values_old + torch.clamp(
                            vpred - mb_values_old,
                            -self.clip_eps, self.clip_eps
                        )
                        v_clipped_loss = (v_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(
                            v_unclipped, v_clipped_loss
                        ).mean()

                        ent_loss = entropy.mean()
                        loss = (policy_loss
                                + self.vf_coef * value_loss
                                - self.ent_coef * ent_loss)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        network.parameters(), max_norm=0.5
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    total_ploss += policy_loss.item()
                    total_vloss += value_loss.item()
                    total_ent += ent_loss.item()
                    n_updates += 1

            self.scheduler.step()

            # ── Entropy annealing: linear decay from start to end ──
            progress = min(ep / max(epochs - 1, 1), 1.0)
            self.ent_coef = (self.ent_coef_start
                             + progress * (self.ent_coef_end - self.ent_coef_start))

            # ── Logging ──
            if ep % log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / max(elapsed, 1e-8)
                rollout_return = rewards_buf.sum(dim=0).mean().item()
                lr = self.scheduler.get_last_lr()[0]
                avg_pl = total_ploss / max(n_updates, 1)
                avg_vl = total_vloss / max(n_updates, 1)
                avg_ent = total_ent / max(n_updates, 1)

                if hasattr(ep_iter, 'set_postfix'):
                    ep_iter.set_postfix({
                        'ret': f"{rollout_return:.1f}",
                        'tile': self.historical_max_tile,
                        'ent': f"{self.ent_coef:.3f}",
                        'fps': f"{fps:,.0f}",
                    })

                self.training_history.append({
                    'epoch': ep,
                    'timesteps': self.total_timesteps,
                    'episodes': self.total_episodes,
                    'policy_loss': round(avg_pl, 4),
                    'value_loss': round(avg_vl, 4),
                    'entropy': round(avg_ent, 4),
                    'max_tile': self.historical_max_tile,
                    'rollout_return': round(rollout_return, 4),
                })

            # ── Checkpoint ──
            if ep % checkpoint_interval == 0 and ep > 0:
                ckpt_path = f"models_2.0/ppo_ep{ep}.pt"
                self.save_model(ckpt_path)

        total_time = time.time() - start_time
        print(f"\n{'=' * 65}")
        print(f"  Done in {total_time:.1f}s | Steps: {self.total_timesteps:,}")
        print(f"  Episodes: {self.total_episodes:,} | "
              f"Max tile: {self.historical_max_tile}")
        print(f"  FPS: {self.total_timesteps / total_time:,.0f}")
        print(f"{'=' * 65}\n")

        if save_path:
            self.save_model(save_path)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grid_size': self.grid_size,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'training_history': self.training_history,
            'historical_max_tile': self.historical_max_tile,
        }, path)
        print(f"  Saved: {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        # Check for architecture mismatch (old model has batch_norm1, new has bn1)
        if any('batch_norm1' in k for k in sd.keys()):
            raise ValueError(
                f"{path} uses the old architecture (log2 encoding). "
                f"Cannot resume — train from scratch instead."
            )
        self.network.load_state_dict(sd)
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.total_timesteps = ckpt.get('total_timesteps', 0)
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.training_history = ckpt.get('training_history', [])
        self.historical_max_tile = ckpt.get('historical_max_tile', 0)
        print(f"  Loaded: {path}")


# ═══════════════════════════════════════════════════════════════════
#  PPO AGENT — BaseAgent wrapper for InteractionModule
# ═══════════════════════════════════════════════════════════════════

class PPOAgent(BaseAgent):
    """Trained PPO agent for evaluation via InteractionModule."""

    def __init__(self, model_path=None, grid_size=4,
                 deterministic=True, device=None):
        super().__init__("PPO")
        self.grid_size = grid_size
        self.deterministic = deterministic

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.network = PPONetworkRef(grid_size=grid_size).to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            else:
                sd = ckpt
            # Check for architecture mismatch (old log2 model vs new one-hot model)
            if any('batch_norm1' in k for k in sd.keys()):
                print(f"WARNING: {model_path} uses the old architecture (log2 encoding).")
                print(f"         The current model uses one-hot encoding.")
                print(f"         Cannot load — using random initialization instead.")
                print(f"         Retrain with: python ppo_agent.py --mode train")
            else:
                self.network.load_state_dict(sd)
                print(f"PPO: loaded {model_path}")
        else:
            print("PPO: randomly initialized (untrained)")

        self.network.eval()

    def choose_action(self, state: np.ndarray,
                      available_moves: List[Action],
                      game_context=None) -> Action:
        device = self.device

        # Build legal mask from available moves
        legal_mask = torch.zeros(1, NUM_ACTIONS, device=device)
        for a in available_moves:
            legal_mask[0, a.value] = 1.0

        board_t = torch.from_numpy(
            state.astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = self.network.actor(board_t, legal_mask)

            if self.deterministic:
                action_idx = torch.argmax(log_probs, dim=1).item()
            else:
                dist = Categorical(logits=log_probs)
                action_idx = dist.sample().item()

        chosen = ACTIONS[action_idx]
        if chosen not in available_moves:
            chosen = available_moves[0]
        return chosen


# ═══════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════

'''def train_and_evaluate(config=None, grid_size=4, num_envs=128,
                       epochs=100000, eval_games=100,
                       save_path="ppo_model.pt"):
    from interaction import InteractionModule

    config = config or {'grid_size': grid_size}
    grid_size = config.get('grid_size', grid_size)

    trainer = PPOTrainer(config=config, num_envs=num_envs, epochs=epochs)
    trainer.train(use_shaping=True, save_path=save_path)

    agent = PPOAgent(model_path=save_path, grid_size=grid_size,
                     deterministic=True)
    module = InteractionModule(config, agent, verbose=True)
    module.set_training_stats(
        training_time_sec=sum(
            h.get('time_sec', 0) for h in trainer.training_history
        ),
        training_episodes=trainer.total_episodes,
    )

    module.run(num_games=eval_games)
    module.print_results()'''
    #module.save_results(f"ppo_{grid_size}x{grid_size}_results.json")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 2048 (game.py Engine)")
    parser.add_argument("--mode", choices=["train", "eval", "full"],
                        default="full")
    parser.add_argument("--model", default="ppo_model.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--shaping", action="store_true", default=True)
    parser.add_argument("--no-shaping", dest="shaping", action="store_false")
    parser.add_argument("--ppo-epochs", type=int, default=3,
                        help="Mini-batch epochs per rollout")
    parser.add_argument("--mini-batch-size", type=int, default=1024)
    parser.add_argument("--rollout-length", type=int, default=64)
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded {args.config} "
              f"(grid: {config.get('grid_size', 4)}x"
              f"{config.get('grid_size', 4)})")
    except FileNotFoundError:
        config = {"grid_size": 4}
        print(f"{args.config} not found, using defaults (4x4)")

    grid_size = config.get("grid_size", 4)

    if args.mode == "train":
        trainer = PPOTrainer(
            config=config, num_envs=args.num_envs, epochs=args.epochs,
            lr=args.lr, ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            rollout_length=args.rollout_length,
        )
        if args.resume:
            trainer.load_model(args.resume)
            print(f"Resuming from {args.resume}")
        trainer.train(use_shaping=args.shaping, save_path=args.model)

    elif args.mode == "eval":
        from interaction import InteractionModule
        from utils import RunLogger
        logger = RunLogger()
        agent = PPOAgent(
            model_path=args.model, grid_size=grid_size, deterministic=True
        )
        module = InteractionModule(config=config, agent=agent, logger=logger, verbose=True, print_board=True)
        module.run(num_games=10)
        module.print_results()

    elif args.mode == "full":
        trainer = PPOTrainer(
            config=config, num_envs=args.num_envs, epochs=args.epochs,
            lr=args.lr, ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            rollout_length=args.rollout_length,
        )
        if args.resume:
            trainer.load_model(args.resume)
        trainer.train(use_shaping=args.shaping, save_path=args.model)

        agent = PPOAgent(
            model_path=args.model, grid_size=grid_size, deterministic=True
        )
        from interaction import InteractionModule
        from utils import RunLogger
        logger = RunLogger()
        module = InteractionModule(config, agent, logger=logger, verbose=True, print_board=True)
        module.set_training_stats(
            training_time_sec=sum(
                h.get('time_sec', 0) for h in trainer.training_history
            ),
            training_episodes=trainer.total_episodes,
        )
        module.run(num_games=args.eval_games)
        module.print_results()
        #module.save_results(f"ppo_{grid_size}x{grid_size}_results.json")