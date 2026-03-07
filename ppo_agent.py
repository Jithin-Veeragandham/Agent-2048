import math
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Game2048, Action
from interaction import BaseAgent
from evaluation import RewardFunction, REWARD_RL


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

NUM_ACTIONS = 4
MAX_TILE_POWER = 17  # 2^17 = 131072
ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


# ═══════════════════════════════════════════════════════════════════
#  STATE ENCODING — same as AlphaZero for consistency
# ═══════════════════════════════════════════════════════════════════

def encode_board(board: np.ndarray) -> np.ndarray:
    """One-hot encode the board over tile powers.

    Channel 0 = empty cells, channel k = cells with value 2^k.
    Shape: (MAX_TILE_POWER + 1, H, W).

    Args:
        board: 2D numpy array of tile values.

    Returns:
        np.ndarray: Float32 tensor of shape (18, H, W).
    """
    rows, cols = board.shape
    encoded = np.zeros((MAX_TILE_POWER + 1, rows, cols), dtype=np.float32)
    encoded[0] = (board == 0).astype(np.float32)

    for r in range(rows):
        for c in range(cols):
            val = board[r, c]
            if val > 0:
                power = int(np.log2(val))
                if power <= MAX_TILE_POWER:
                    encoded[power][r][c] = 1.0
    return encoded


# ═══════════════════════════════════════════════════════════════════
#  NEURAL NETWORK — actor-critic with shared backbone
# ═══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block: conv → BN → ReLU → conv → BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO.

    Shared CNN trunk splits into:
        - Actor head:  outputs log-probabilities over 4 actions
        - Critic head: outputs scalar state value V(s)

    The shared trunk lets both heads benefit from the same
    learned features, which is more parameter-efficient than
    having two completely separate networks.

    Why actor-critic?
        - Actor alone (REINFORCE): high variance, slow learning
        - Critic alone (DQN): no direct policy, struggles with
          discrete action selection
        - Actor-critic: critic reduces variance of actor's updates,
          actor gives a direct stochastic policy

    Args:
        grid_size: Board dimension (6 for 6×6).
        num_res_blocks: Depth of the shared trunk.
        channels: Width of convolutional layers.
    """

    def __init__(self, grid_size: int = 6, num_res_blocks: int = 4,
                 channels: int = 128):
        super().__init__()
        self.grid_size = grid_size
        input_channels = MAX_TILE_POWER + 1

        # ── Shared trunk ──
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *[ResidualBlock(channels) for _ in range(num_res_blocks)],
        )

        flat_size = channels * grid_size * grid_size

        # ── Actor head ── (outputs action probabilities)
        # "Given this board, which move should I make?"
        self.actor = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

        # ── Critic head ── (outputs state value)
        # "How good is this board position overall?"
        self.critic = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * grid_size * grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through shared trunk + both heads.

        Args:
            x: Encoded board, shape (batch, channels, H, W).

        Returns:
            Tuple of:
                - action_logits: shape (batch, 4) — raw logits
                - state_value: shape (batch, 1) — V(s) estimate
        """
        features = self.trunk(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self, board: np.ndarray, available_moves: List[Action],
        deterministic: bool = False,
    ) -> Tuple[int, float, float, float]:
        """Convenience: board → (action, log_prob, entropy, value).

        Masks invalid moves before sampling. Used during rollout
        collection and at inference time.

        Args:
            board: Raw 2D numpy board.
            available_moves: List of valid Action values.
            deterministic: If True, pick argmax instead of sampling.

        Returns:
            Tuple of:
                - action_idx: int (0–3)
                - log_prob: log probability of the chosen action
                - entropy: policy entropy (for exploration bonus)
                - value: V(s) estimate
        """
        self.eval()
        device = next(self.parameters()).device

        encoded = encode_board(board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value = self.forward(tensor)

        # Mask invalid actions → set to -inf so softmax gives 0
        available_idx = set(a.value for a in available_moves)
        mask = torch.full((1, NUM_ACTIONS), float('-inf'), device=device)
        for i in available_idx:
            mask[0, i] = 0.0
        masked_logits = logits + mask

        # Sample from masked distribution
        probs = F.softmax(masked_logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (
            action.item(),
            log_prob.item(),
            entropy.item(),
            value.item(),
        )


# ═══════════════════════════════════════════════════════════════════
#  ROLLOUT BUFFER — stores trajectories for on-policy learning
# ═══════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores a batch of experience for PPO training.

    PPO is ON-POLICY: it can only learn from data collected by the
    CURRENT policy. Once you update the network, old data is stale
    and must be thrown away. This buffer collects one "rollout"
    (a batch of steps across potentially multiple games), then
    gets consumed by the training step and cleared.

    This is fundamentally different from DQN's replay buffer:
        - DQN replay buffer: stores old data, samples randomly, persists
        - PPO rollout buffer: stores current data, used once, cleared

    Stores per-step:
        - state: encoded board
        - action: chosen action index
        - log_prob: log probability under the policy that chose it
        - reward: shaped reward from the reward function
        - value: critic's V(s) estimate
        - done: whether the game ended after this step
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, state: np.ndarray, action: int, log_prob: float,
            reward: float, value: float, done: bool):
        """Add one timestep of experience."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    @property
    def size(self) -> int:
        return len(self.states)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        GAE (Generalized Advantage Estimation) balances bias vs variance:
            - lambda=0: just TD error (low variance, high bias)
            - lambda=1: full Monte Carlo return (high variance, low bias)
            - lambda=0.95: sweet spot used by most PPO implementations

        The advantage A(s,a) answers: "was this action BETTER or WORSE
        than what the critic expected?" Positive = good surprise,
        negative = bad surprise.

        Args:
            last_value: V(s) for the state after the last step
                (needed because the rollout may end mid-game).
            gamma: Discount factor. 0.99 means future rewards are worth
                slightly less than immediate ones.
            gae_lambda: GAE smoothing parameter.

        Returns:
            Tuple of:
                - returns: discounted return for each step
                - advantages: GAE advantage for each step (normalized)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD error: how much better was the actual outcome than predicted?
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

            # GAE: exponentially weighted sum of TD errors
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values

        # Normalize advantages (zero mean, unit variance)
        # This stabilizes training by ensuring updates aren't dominated
        # by the scale of the advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def get_tensors(self, device: str = 'cpu'
                    ) -> Tuple[torch.Tensor, ...]:
        """Convert buffer contents to PyTorch tensors.

        Returns:
            Tuple of (states, actions, old_log_probs) as tensors.
        """
        states = torch.from_numpy(
            np.stack([encode_board(s) for s in self.states])
        ).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(
            self.log_probs, dtype=torch.float32, device=device
        )
        return states, actions, old_log_probs

    def clear(self):
        """Discard all stored data. Called after each training update."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


# ═══════════════════════════════════════════════════════════════════
#  PPO TRAINER
# ═══════════════════════════════════════════════════════════════════

class PPOTrainer:
    """Training loop for PPO on 2048.

    The training cycle:
        1. COLLECT: play for N steps using current policy, store transitions
        2. COMPUTE: calculate GAE advantages and returns
        3. UPDATE: run K epochs of mini-batch gradient descent with
           the clipped surrogate objective
        4. CLEAR: throw away old data (on-policy requirement)
        5. REPEAT

    The clipped objective is the heart of PPO:
        L = min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)

    where r(θ) = π_new(a|s) / π_old(a|s) is the probability ratio.

    If the advantage A is positive (good action):
        - r(θ) > 1+ε: policy moved too far toward this action → clipped
        - This prevents overcommitting to actions that happened to work

    If the advantage A is negative (bad action):
        - r(θ) < 1-ε: policy moved too far away → clipped
        - This prevents overcorrecting away from bad actions

    The clip keeps updates small and safe — you can never change
    the policy by more than ε per update.

    Args:
        grid_size: Board dimension.
        num_res_blocks: Residual blocks in network.
        channels: Conv channels in network.
        lr: Learning rate.
        gamma: Discount factor for future rewards.
        gae_lambda: GAE smoothing parameter.
        clip_epsilon: PPO clipping range (ε).
        value_coef: Weight of value loss in total loss.
        entropy_coef: Weight of entropy bonus (encourages exploration).
        rollout_steps: Steps to collect before each training update.
        num_epochs: Mini-batch epochs per training update.
        batch_size: Mini-batch size.
        reward_fn: RewardFunction for board evaluation.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        grid_size: int = 6,
        num_res_blocks: int = 4,
        channels: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        rollout_steps: int = 2048,
        num_epochs: int = 4,
        batch_size: int = 64,
        reward_fn: Optional[RewardFunction] = None,
        device: str = None,
    ):
        # Load full config — grid_size param is fallback if no config given
        self.config = config or {'grid_size': grid_size}
        self.grid_size = self.config.get('grid_size', grid_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Reward function — simplified for RL (let network learn structure)
        self.reward_fn = reward_fn or REWARD_RL

        # Network + optimizer
        self.network = PPONetwork(
            grid_size=self.grid_size,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr,
                                    eps=1e-5)

        # Rollout storage
        self.buffer = RolloutBuffer()

        # Training statistics
        self.training_history: List[Dict] = []
        self.total_timesteps: int = 0
        self.total_episodes: int = 0

        # Current game state (persists across rollout collections)
        self._game: Optional[Game2048] = None
        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._recent_scores: List[int] = []

    # ─── Reward Shaping ───────────────────────────────────────────

    def _compute_reward(self, old_board: np.ndarray, new_board: np.ndarray,
                        merge_reward: int, done: bool) -> float:
        """Compute the shaped reward for a single transition.

        Combines:
            - Merge reward from the game (raw score gained this move)
            - Change in board evaluation from the reward function
            - Death penalty if the game just ended

        This is different from the evaluation module's reward function
        (which evaluates a board state). Here we compute a TRANSITION
        reward: how much better did the board get from this move?

        Args:
            old_board: Board before the move.
            new_board: Board after the move + tile spawn.
            merge_reward: Raw points from merges this move.
            done: Whether the game ended.

        Returns:
            float: Shaped reward for this transition.
        """
        # Component 1: raw merge reward (normalized)
        merge_component = merge_reward / 100.0

        # Component 2: improvement in board quality
        old_eval = self.reward_fn.compute(old_board)
        new_eval = self.reward_fn.compute(new_board)
        eval_delta = new_eval - old_eval

        # Component 3: death penalty
        death_penalty = -5.0 if done else 0.0

        return merge_component + eval_delta * 10.0 + death_penalty

    # ─── Rollout Collection ───────────────────────────────────────

    def _ensure_game(self):
        """Initialize or reset the game when needed."""
        if self._game is None or self._game.is_game_over():
            if self._game is not None:
                self._recent_scores.append(self._game.get_score())
                if len(self._recent_scores) > 100:
                    self._recent_scores = self._recent_scores[-100:]
                self.total_episodes += 1

            self._game = Game2048(self.config)
            self._episode_reward = 0.0
            self._episode_length = 0

    def collect_rollout(self) -> float:
        """Collect rollout_steps of experience using the current policy.

        Plays the game step by step, storing (s, a, log_prob, r, v, done)
        into the buffer. Automatically resets the game when it ends
        and starts a new one.

        Returns:
            float: Last value estimate (for GAE bootstrapping).
        """
        self.buffer.clear()
        self.network.eval()

        for _ in range(self.rollout_steps):
            self._ensure_game()

            state = self._game.get_state()
            available = self._game.get_available_moves()

            if not available:
                # Game over mid-step
                self._game = None
                self._ensure_game()
                state = self._game.get_state()
                available = self._game.get_available_moves()

            # Agent decides
            action_idx, log_prob, entropy, value = (
                self.network.get_action_and_value(state, available)
            )

            # Execute action
            old_board = state.copy()
            action = ACTIONS[action_idx]
            valid, merge_reward = self._game.move(action)

            if not valid:
                # Shouldn't happen due to masking, but handle gracefully
                # Pick a random valid move instead
                action = available[np.random.randint(len(available))]
                action_idx = action.value
                valid, merge_reward = self._game.move(action)

            new_board = self._game.get_state()
            done = self._game.is_game_over()

            # Shaped reward
            reward = self._compute_reward(old_board, new_board,
                                          merge_reward, done)

            self.buffer.add(state, action_idx, log_prob,
                            reward, value, done)

            self._episode_reward += reward
            self._episode_length += 1
            self.total_timesteps += 1

        # Bootstrap value for the last state
        # (needed because rollout might end mid-game)
        self._ensure_game()
        state = self._game.get_state()
        available = self._game.get_available_moves()
        if available:
            _, _, _, last_value = self.network.get_action_and_value(
                state, available
            )
        else:
            last_value = 0.0

        return last_value

    # ─── PPO Update ───────────────────────────────────────────────

    def update(self, last_value: float) -> Dict[str, float]:
        """Run the PPO update step on the collected rollout.

        Steps:
            1. Compute GAE advantages and returns
            2. For K epochs, iterate over mini-batches:
                a. Forward pass: get new log_probs and values
                b. Compute probability ratio: r = exp(new_log_prob - old_log_prob)
                c. Clipped surrogate: min(r*A, clip(r)*A)
                d. Value loss: MSE(predicted_V, returns)
                e. Entropy bonus: encourage exploration
                f. Total loss = -policy_loss + value_coef*value_loss - entropy_coef*entropy
                g. Backprop and step

        Args:
            last_value: Bootstrap value for GAE computation.

        Returns:
            Dict with 'policy_loss', 'value_loss', 'entropy', 'clip_fraction'.
        """
        # Compute advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        returns_tensor = torch.tensor(returns, dtype=torch.float32,
                                       device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32,
                                          device=self.device)

        states, actions, old_log_probs = self.buffer.get_tensors(self.device)

        # Training
        self.network.train()
        n = self.buffer.size
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            # Shuffle indices for mini-batch sampling
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns_tensor[batch_idx]
                b_advantages = advantages_tensor[batch_idx]

                # Forward pass with current network
                logits, values = self.network(b_states)
                values = values.squeeze(-1)

                # Get new log probs for the actions that were taken
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # ── Core PPO: clipped surrogate objective ──
                # Probability ratio: how much has the policy changed?
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                # Unclipped objective
                surr1 = ratio * b_advantages

                # Clipped objective: cap the ratio at [1-ε, 1+ε]
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                )
                surr2 = clipped_ratio * b_advantages

                # Take the MINIMUM — pessimistic bound
                # If advantage > 0: prevents ratio from going above 1+ε
                #   (don't overcommit to good actions)
                # If advantage < 0: prevents ratio from going below 1-ε
                #   (don't overcorrect away from bad actions)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: MSE between critic and actual returns
                value_loss = F.mse_loss(values, b_returns)

                # Total loss: policy + value - entropy bonus
                # Entropy bonus encourages exploration by penalizing
                # overconfident policies
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

                # Clip fraction: how often was the ratio actually clipped?
                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon)
                        .float().mean().item()
                    )
                total_clip_frac += clip_frac
                num_updates += 1

        return {
            'policy_loss': round(total_policy_loss / max(num_updates, 1), 4),
            'value_loss': round(total_value_loss / max(num_updates, 1), 4),
            'entropy': round(total_entropy / max(num_updates, 1), 4),
            'clip_fraction': round(total_clip_frac / max(num_updates, 1), 4),
        }

    # ─── Main Training Loop ───────────────────────────────────────

    def train(self, total_timesteps: int = 500_000,
              log_interval: int = 10, save_path: Optional[str] = None):
        """Full PPO training pipeline.

        Each iteration:
            1. Collect rollout_steps of experience
            2. Compute advantages
            3. Run PPO update (K epochs of mini-batch SGD)
            4. Log progress

        Args:
            total_timesteps: Total environment steps for training.
            log_interval: Print stats every N iterations.
            save_path: Save model after training (and periodically).
        """
        num_updates = total_timesteps // self.rollout_steps

        print(f"\n{'═' * 60}")
        print(f"  PPO Training — {self.grid_size}×{self.grid_size}")
        print(f"  {total_timesteps} total steps, "
              f"{self.rollout_steps} steps/rollout, "
              f"{num_updates} updates")
        print(f"  Clip ε={self.clip_epsilon}, γ={self.gamma}, "
              f"λ={self.gae_lambda}")
        print(f"  Device: {self.device}")
        print(f"{'═' * 60}\n")

        start_time = time.time()

        for update_idx in range(num_updates):
            # Collect experience
            last_value = self.collect_rollout()

            # PPO update
            losses = self.update(last_value)

            # Logging
            self.training_history.append({
                'update': update_idx + 1,
                'timesteps': self.total_timesteps,
                'episodes': self.total_episodes,
                **losses,
            })

            if (update_idx + 1) % log_interval == 0 or update_idx == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / max(elapsed, 1e-8)
                avg_score = (
                    np.mean(self._recent_scores) if self._recent_scores
                    else 0.0
                )
                max_score = (
                    max(self._recent_scores) if self._recent_scores
                    else 0
                )

                print(
                    f"  Update {update_idx + 1:>5}/{num_updates}  |  "
                    f"Steps: {self.total_timesteps:>8}  |  "
                    f"Episodes: {self.total_episodes:>5}  |  "
                    f"Avg Score: {avg_score:>8.0f}  |  "
                    f"Max Score: {max_score:>7}  |  "
                    f"P_loss: {losses['policy_loss']:.3f}  "
                    f"V_loss: {losses['value_loss']:.3f}  "
                    f"Ent: {losses['entropy']:.3f}  "
                    f"Clip: {losses['clip_fraction']:.2f}  |  "
                    f"FPS: {fps:.0f}"
                )

        total_time = time.time() - start_time
        print(f"\n{'═' * 60}")
        print(f"  Training complete in {total_time:.1f}s")
        print(f"  Total steps: {self.total_timesteps}")
        print(f"  Total episodes: {self.total_episodes}")
        if self._recent_scores:
            print(f"  Final avg score (last 100): "
                  f"{np.mean(self._recent_scores):.0f}")
        print(f"{'═' * 60}\n")

        if save_path:
            self.save_model(save_path)

    # ─── Save / Load ──────────────────────────────────────────────

    def save_model(self, path: str):
        """Save network weights, optimizer state, and training history.

        Args:
            path: File path (e.g. 'ppo_6x6.pt').
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grid_size': self.grid_size,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'training_history': self.training_history,
        }, path)
        print(f"  Model saved to {path}")

    def load_model(self, path: str):
        """Load network weights from a checkpoint.

        Args:
            path: Path to a saved .pt file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.training_history = checkpoint.get('training_history', [])
        print(f"  Model loaded from {path}")


# ═══════════════════════════════════════════════════════════════════
#  AGENT — plugs into InteractionModule via BaseAgent
# ═══════════════════════════════════════════════════════════════════

class PPOAgent(BaseAgent):
    """PPO agent implementing BaseAgent for the Interaction Module.

    Once trained, PPO is extremely fast at decision time — just a
    single forward pass through the network, no search needed.
    This makes it the fastest agent at inference among all five
    in your project.

    Args:
        model_path: Path to trained .pt checkpoint. If None, uses
            random weights (for testing).
        grid_size: Board dimension. Must match trained model.
        deterministic: If True, always pick the highest-probability
            action. If False, sample from the distribution.
        device: 'cuda' or 'cpu'.

    Usage::

        agent = PPOAgent(model_path="ppo_6x6.pt", grid_size=6)
        module = InteractionModule(config, agent)
        module.run(num_games=100)
        module.print_results()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        grid_size: int = 6,
        deterministic: bool = True,
        device: str = None,
    ):
        super().__init__("PPO")
        self.grid_size = grid_size
        self.deterministic = deterministic
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = PPONetwork(grid_size=grid_size).to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print(f"PPO: loaded model from {model_path}")
        else:
            print("PPO: using randomly initialized network (untrained)")

        self.network.eval()

    def choose_action(
        self,
        state: np.ndarray,
        available_moves: List[Action],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """Pick a move via single forward pass — no search needed.

        This is why PPO has the fastest inference time among all agents.
        DQN is similarly fast. MCTS, Beam Search, and AlphaZero all
        require expensive search at decision time.

        Args:
            state: Current board.
            available_moves: Valid moves.
            game_context: Not used by PPO.

        Returns:
            Action: The chosen move.
        """
        action_idx, _, _, _ = self.network.get_action_and_value(
            state, available_moves, deterministic=self.deterministic
        )

        chosen = ACTIONS[action_idx]

        # Safety fallback
        if chosen not in available_moves:
            chosen = available_moves[0]

        return chosen


# ═══════════════════════════════════════════════════════════════════
#  CONVENIENCE — train + evaluate in one script
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(
    config: Optional[Dict] = None,
    grid_size: int = 6,
    total_timesteps: int = 500_000,
    eval_games: int = 100,
    save_path: str = "ppo_model.pt",
):
    """Complete pipeline: train PPO, then evaluate it.

    Args:
        config: Full game config dict (from config.json). If None,
            uses grid_size to build a minimal config.
        grid_size: Board dimension (fallback if config not provided).
        total_timesteps: Total training environment steps.
        eval_games: Number of evaluation games.
        save_path: Where to save the trained model.
    """
    from interaction import InteractionModule

    config = config or {'grid_size': grid_size}
    grid_size = config.get('grid_size', grid_size)

    # ── Train ──
    trainer = PPOTrainer(config=config)
    trainer.train(total_timesteps=total_timesteps, save_path=save_path)

    # ── Evaluate ──
    agent = PPOAgent(
        model_path=save_path,
        grid_size=grid_size,
        deterministic=True,
    )

    module = InteractionModule(config, agent, verbose=True)

    # Record training cost for efficiency metrics
    total_train_time = sum(
        h.get('time_sec', 0) for h in trainer.training_history
    )
    module.set_training_stats(
        training_time_sec=total_train_time if total_train_time > 0
        else time.time(),
        training_episodes=trainer.total_episodes,
    )

    module.run(num_games=eval_games)
    module.print_results()
    module.save_results(f"ppo_{grid_size}x{grid_size}_results.json")


# ═══════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 2048 Agent")
    parser.add_argument("--mode", choices=["train", "eval", "full"],
                        default="full", help="train / eval / full pipeline")
    parser.add_argument("--model", default="ppo_model.pt",
                        help="Model checkpoint path")
    parser.add_argument("--config", default="config.json",
                        help="Path to game config JSON file")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of evaluation games")

    args = parser.parse_args()

    # Load config.json
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded {args.config} (grid: {config.get('grid_size', 4)}×"
              f"{config.get('grid_size', 4)})")
    except FileNotFoundError:
        config = {"grid_size": 6}
        print(f"Config {args.config} not found, using defaults (6×6)")

    grid_size = config.get("grid_size", 6)

    if args.mode == "train":
        trainer = PPOTrainer(config=config)
        trainer.train(total_timesteps=args.timesteps, save_path=args.model)

    elif args.mode == "eval":
        from interaction import InteractionModule
        agent = PPOAgent(
            model_path=args.model,
            grid_size=grid_size,
            deterministic=True,
        )
        module = InteractionModule(config, agent, verbose=True)
        module.run(num_games=args.eval_games)
        module.print_results()

    elif args.mode == "full":
        train_and_evaluate(
            config=config,
            total_timesteps=args.timesteps,
            eval_games=args.eval_games,
            save_path=args.model,
        )