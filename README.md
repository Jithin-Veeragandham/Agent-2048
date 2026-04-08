# 2048 AI Research Platform

A research platform for benchmarking AI search and reinforcement learning algorithms on the 2048 game. The project compares four algorithm families — **Beam Search**, **MCTS**, **Expectimax**, **AlphaZero**, and **PPO** — across win rate, average score, and compute cost, to evaluate the trade-off between search quality and inference time.

**Input:** a 4×4 board state and a set of valid moves.  
**Output:** a chosen action, a structured game record (score, tile distribution, per-move reward breakdown), and evaluation plots.

Each algorithm is a self-contained agent that plugs into a shared harness (`InteractionModule`). Changing the agent under test is a one-line swap; the game, logging, and evaluation layers are unaffected.

---

## Project Structure

```
2048-ai/
├── game/                       # Game engine (pure logic + pygame renderer)
│   ├── __init__.py
│   └── engine.py               # Game2048, Game2048Visual, Action
│
├── agents/                     # AI algorithms
│   ├── __init__.py
│   ├── base.py                 # BaseAgent ABC — all agents implement this
│   ├── beam_search.py          # Heuristic Beam Search
│   ├── mcts.py                 # Classic MCTS + Heuristic MCTS
│   ├── expectimax.py           # Expectimax search
│   ├── expectimax_snake.py     # Expectimax with snake-pattern heuristic
│   ├── alphazero.py            # AlphaZero (MCTS + neural network)
│   └── ppo2048.py              # PPO (policy gradient)
│
├── framework/                  # Experiment harness + evaluation
│   ├── __init__.py
│   ├── interaction.py          # InteractionModule — runs episodes, logs results
│   ├── evaluation.py           # RewardFunction + GameEvaluator
│   └── logger.py               # RunLogger — JSONL logging per episode
│
├── visualization/              # Live agent visualizer
│   ├── agent_viewer.py         # Watch one agent play in real time (pygame)
│   └── multi_viewer.py         # Watch all agents race side by side
│
├── scripts/                    # Standalone entry-point scripts
│   ├── pretrain_alphazero.py   # Pretrain AlphaZero on ExpectimaxSnake demos
│   ├── eval_pretrained.py      # Evaluate pretrained network (greedy, no MCTS)
│   └── cma_tuner.py            # CMA-ES reward weight tuner for Beam Search
│
├── analysis/                   # Post-hoc analysis scripts
│   ├── analyze_all.py          # Full evaluation + strategy plots for all agents
│   ├── evaluate_agents.py      # Summary table + score distributions
│   ├── strategy.py             # Per-agent win/loss strategy breakdown
│   └── heuristic_trajectory.py # Per-move reward component trajectory plots
│
├── checkpoints/                # Saved model checkpoints (AlphaZero, PPO)
├── tests/                      # Unit test suite (pytest)
├── results/                    # Generated plots (gitignored)
├── logs/                       # Run logs (JSONL, gitignored)
├── config.json                 # Default game configuration
├── pytest.ini                  # Test runner configuration
└── requirements.txt
```

---

## Setup

All commands must be run from the **project root** (the directory containing `config.json`). Imports like `from game import` and `from agents import` rely on the working directory being the root.

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
pip install pytest           # for running tests
```

---

## Quick Start

### Run a single agent

```python
from game import Game2048
from agents import BeamSearchAgent
from framework.interaction import InteractionModule
from framework.logger import RunLogger

agent  = BeamSearchAgent(beam_width=10, search_depth=15)
logger = RunLogger()

module = InteractionModule(
    config={"grid_size": 4},
    agent=agent,
    logger=logger,
    verbose=True,
)
module.run(num_games=100)
module.print_results()
```

### Run with parallel workers (game-level parallelism)

```python
module = InteractionModule(
    config={"grid_size": 4},
    agent=agent,
    num_workers=8,   # 8 games in parallel via ProcessPoolExecutor
)
module.run(num_games=100)
```

### Compare multiple agents

```python
from framework.interaction import run_comparison
from agents import BeamSearchAgent, MCTSAgent

run_comparison(
    config={"grid_size": 4},
    agents=[BeamSearchAgent(beam_width=10), MCTSAgent(num_simulations=300)],
    num_games=100,
    num_workers=8,
    verbose=True,
)
```

### Human play

```bash
python game/engine.py             # play with arrow keys / WASD
```

---

## Results

Performance across 100 games per agent (except AlphaZero — still training):

| Agent | Avg Score | Win Rate (2048+) | Best Tile |
|---|---|---|---|
| ExpectimaxSnake (d=2) | 68,770 | 93% | 8192 |
| PPO | 37,799 | 82% | 4096 |
| MCTS (n=300, d=15) | 26,629 | 66% | 4096 |
| BeamSearch (w=10, d=15) | 26,429 | 67% | 2048 |
| AlphaZero | — | — | — |

AlphaZero is excluded — the checkpoint in `checkpoints/` represents an early training run (62 episodes) and is not representative. Competitive performance requires significantly more self-play. See the [AlphaZero section](#alphazero) for context.

---

## Agents

### Beam Search

Breadth-limited tree search that expands all four moves at each depth level, scores every resulting board with the composite `RewardFunction`, and keeps only the top `beam_width` states to expand next. At the end of the search horizon, the action leading to the highest-scoring branch is chosen.

**Params used:** `beam_width=10, search_depth=15`

**Architecture choices:**
- The heuristic is the six-component `RewardFunction` (`tile_score`, `empty_bonus`, `monotonicity`, `corner_bonus`, `merge_potential`, `smoothness`). Getting the component weights right required significant research — raw merge score is too myopic, and each component was chosen to capture a distinct structural property of a "good" board position.
- Weight tuning is automated via **CMA-ES** (`scripts/cma_tuner.py`), which treats the six weights as a continuous optimisation problem and searches for the combination that maximises average score over a batch of games. Best weights are saved to `logs/cma_best_weights.json`.

```python
from agents import BeamSearchAgent
agent = BeamSearchAgent(beam_width=10, search_depth=15)
```

```bash
python scripts/cma_tuner.py            # fresh CMA-ES run
python scripts/cma_tuner.py --resume   # resume from logs/cma_checkpoint.json
```

---

### MCTS (Classic)

Standard UCT: builds a search tree using UCB1 to balance exploration vs exploitation, runs random rollouts from leaf nodes to estimate value, and backpropagates game scores up the tree.

**Params used:** `num_simulations=300, rollout_depth=15`

**Architecture choices:**
- 2048 scores can reach tens of thousands. Without normalisation the exploitation term in UCB1 completely dominates and MCTS degenerates to a greedy policy. Running **min/max normalisation** across all visited nodes — rescaling observed rewards to [0, 1] before applying UCB1 — was a significant improvement in win rate.
- A precomputed and import-time cached **snake-weight matrix** (4×4 grid of powers of 4 in zigzag order) is available for rollout guidance in the heuristic variant below.

```python
from agents import MCTSAgent
agent = MCTSAgent(num_simulations=300, rollout_depth=15)
```

---

### MCTS (Heuristic)

Same UCT tree structure as Classic MCTS, but rollouts use the snake-weight heuristic to guide move selection rather than picking randomly. The key design decision: the **heuristic guides rollout move selection**, but the **value backpropagated is the raw game score** — not the heuristic value. This gives UCB1 large, well-separated values to work with while the heuristic steers simulations toward good board states.

**Params used:** `num_simulations=200, rollout_depth=10`

The lower simulation count compared to Classic MCTS reflects the higher per-simulation cost — heuristic evaluation at each rollout step adds overhead, so fewer total simulations fit within the same time budget.

```python
from agents import MCTSHeuristicAgent
agent = MCTSHeuristicAgent(num_simulations=200, rollout_depth=10)
```

---

### Expectimax

Standard expectimax tree with alternating MAX nodes (agent's turn) and CHANCE nodes (random tile spawn). The chance node averages over all possible tile placements weighted by their probability. Included as a baseline — the Snake variant below outperforms it significantly by replacing the raw score evaluation with a structured positional heuristic.

**Params used:** `depth=3`

**Architecture choices:**
- Uses the composite `RewardFunction` to score leaf nodes, giving it the same heuristic signal as Beam Search but with a proper chance-node expectation rather than ignoring tile spawns entirely.
- Depth is limited to 3 in practice — the branching factor (4 moves × 16 possible spawn positions) makes deeper search exponentially expensive.

```python
from agents import ExpectimaxAgent
agent = ExpectimaxAgent(depth=3)
```

---

### Expectimax (Snake heuristic)

The primary Expectimax variant used in evaluations. Uses an **integer/float depth trick** to alternate CHANCE (integer depth) and MAX (float depth) nodes by decrementing depth by 0.5 per level — `depth=2` gives 3 chance levels and 2 player levels.

**Params used:** `depth=2`

**Architecture choices:**
- Evaluation uses `PERFECT_SNAKE`, a fixed positional weight matrix of increasing powers of 2 in a zigzag pattern (top-left = highest weight). The board score is a dot product of tile values against this matrix, encouraging the largest tile to anchor in the top-left and tiles to decrease outward.
- Board operations inside the search are **pure numpy** (no `Game2048` cloning), avoiding Python object allocation overhead per node. Only 2-tiles are spawned in chance nodes.

```python
from agents import ExpectimaxSnakeAgent
agent = ExpectimaxSnakeAgent(depth=2)
```

---

### PPO

Proximal Policy Optimisation with a convolutional actor-critic network. Runs N environments in parallel for batched rollout collection and GPU-accelerated training.

**Architecture:**
- **Input encoding:** tile values are converted to one-hot exponent channels → `(N, 16, 4, 4)` tensor (16 channels for tile exponents 0–15, covering tiles up to 32768)
- **Shared conv backbone:** 2× `Conv2d(→64, 3×3, pad=1) + BatchNorm + ReLU`
- **Actor head:** `FC(1024 → 256) → FC(256 → 64) → FC(64 → 4)` with **legal action masking** (invalid move logits set to `-inf` before softmax)
- **Critic head:** same backbone, separate `FC(→ 1)` value output
- CuDNN autotuner and TF32 enabled for training throughput

Training and evaluation:

```bash
# Train from scratch (128 parallel envs, GPU if available)
python agents/ppo2048.py --mode train --num-envs 128 --epochs 100000 --model checkpoints/ppo_model.pt

# Resume from checkpoint
python agents/ppo2048.py --mode train --resume checkpoints/ppo_model.pt --epochs 50000

# Evaluate a trained checkpoint
python agents/ppo2048.py --mode eval --model checkpoints/ppo_model.pt --eval-games 100
```

```python
from agents.ppo2048 import PPO2048Agent
agent = PPO2048Agent(model_path="checkpoints/ppo_model.pt")
```

---

### AlphaZero

Combines MCTS with a neural network that provides both a **policy prior** (action probabilities) and a **value estimate** (expected return). The network is trained via self-play: MCTS produces improved policy targets, actual game returns provide value targets.

**Architecture:**
- **Input encoding:** same one-hot exponent scheme as PPO → `(B, 16, 4, 4)`
- **Backbone:** 2× `Conv2d(16→128, 3×3, pad=1) + BatchNorm + ReLU` → `FC(2048 → 256)`
- **Policy head:** `FC(256 → 4)` → log_softmax
- **Value head:** `FC(256 → 1)` → tanh, scaled by `V_SCALE=30000` to normalise MC returns to roughly `[-1, 1]`
- **Legal action masking** applied throughout MCTS and at inference
- **No subtree reuse** — random tile spawns make cached subtrees stale between moves
- **Chance node sampling:** at each MCTS leaf, N tile placements are sampled and the network value is averaged, giving a stable expectation over random spawns
- Runs entirely on **CPU** — MCTS is sequential Python tree traversal and GPU transfer overhead on `(1, 16, 4, 4)` tensors exceeds any compute savings

**Pretraining:** to address the cold-start problem, the network can be pretrained on `ExpectimaxSnake(d=2)` game records before self-play begins (`scripts/pretrain_alphazero.py`). This gives the network a meaningful starting point but does not fully solve convergence — competitive performance still requires millions of self-play episodes.

```bash
# Pretrain on ExpectimaxSnake demonstrations
python scripts/pretrain_alphazero.py --save checkpoints/alphazero_pretrained.pt

# Train (fresh or resume from checkpoint)
python agents/alphazero.py --mode train --n_ep 500 --n_mcts 100 --fresh
python agents/alphazero.py --mode train --n_ep 500 --n_mcts 100

# Evaluate
python agents/alphazero.py --mode eval --model checkpoints/alphazero_best.pt --num_games 10
```

```python
from agents import AlphaZeroAgent
agent = AlphaZeroAgent(model_path="checkpoints/alphazero_best.pt", n_mcts=100)
```

---

### Adding a new agent

Subclass `BaseAgent` and implement `choose_action`. The agent is automatically compatible with `InteractionModule`, `RunLogger`, and all analysis scripts:

```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    agent_type = "my_agent"   # determines log filename: logs/my_agent_runs.jsonl

    def __init__(self):
        super().__init__("MyAgent")

    def choose_action(self, state, available_moves, game_context=None):
        # game_context['game']      → clonable Game2048 for lookahead
        # game_context['reward_fn'] → RewardFunction for board evaluation
        return available_moves[0]

    def get_params(self):
        return {}   # logged alongside every game record
```

---

## Reward Function

Six-component composite heuristic used for both evaluation and rollout guidance:

| Component | Role |
|---|---|
| `tile_score` | Weights higher tiles exponentially (value × log₂(value)) |
| `empty_bonus` | Rewards open + clustered empty cells |
| `monotonicity` | Rewards decreasing gradient from corner (snake pattern) |
| `corner_bonus` | Rewards max tile in top-left corner |
| `merge_potential` | Rewards adjacent matching tile pairs, weighted by tile value |
| `smoothness` | Penalises large value gaps between neighbours |

```python
from framework.evaluation import RewardFunction

rf = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.5, 'mono': 2.5,
    'corner': 1.5, 'merge': 0.5, 'smooth': 0.1,
})
score = rf.compute(board)
breakdown = rf.compute_breakdown(board)   # per-component dict
```

To use the best weights found by CMA-ES:

```python
import json
from framework.evaluation import RewardFunction

with open("logs/cma_best_weights.json") as f:
    best = json.load(f)

rf = RewardFunction(weights=best["weights"])
agent = BeamSearchAgent(beam_width=10, search_depth=15, reward_fn=rf)
```

---

## Logging

Each `run()` call appends one JSON line per game to `logs/<agent_type>_runs.jsonl`. The latest run's full per-move detail is written to `logs/latest_run.json`.

```json
{
  "run_id": "abc123def456",
  "game_index": 0,
  "agent": "BeamSearch(w=10,d=15)",
  "agent_params": {"beam_width": 10, "search_depth": 15},
  "score": 34560,
  "highest_tile": 2048,
  "moves": 1243,
  "won": true,
  "reached_2048": true,
  "time_sec": 18.4,
  "avg_inference_ms": 14.8,
  "avg_reward_breakdown": {"tile_score": 0.82, "empty_bonus": 0.31, ...},
  "final_reward_breakdown": {"tile_score": 0.94, ...},
  "quartile_reward_breakdowns": {"q25": {...}, "q50": {...}, "q75": {...}, "q100": {...}}
}
```

Reading logs back:

```python
from framework.logger import load_all_runs, summarize_games

games = load_all_runs("beam_search")
summary = summarize_games(games)
print(f"Win rate: {summary['win_rate_2048']}%")

# Filter to a specific config
bs10 = [g for g in games if g['agent_params'].get('beam_width') == 10]
```

Enable full per-move detail (files ~400× larger):

```python
logger = RunLogger(log_move_detail=True)
```

---

## Visualization

### Single agent

Watch one agent play in real time:

```bash
python visualization/agent_viewer.py --agent beam_search --width 10 --depth 15 --delay 0
python visualization/agent_viewer.py --agent mcts --simulations 300 --delay 200
python visualization/agent_viewer.py --agent expectimax_snake --search-depth 2
python visualization/agent_viewer.py --agent random
```

**Controls:** `SPACE` pause/resume · `R` reset · `Q` quit

Available `--agent` values: `beam_search`, `mcts`, `expectimax`, `expectimax_snake`, `random`

### All agents side by side

Watch all agents race simultaneously in a single tiled window. Each agent runs in its own thread at its natural speed — the compute/quality trade-off between Random (instant) and MCTS (simulation-heavy) is immediately visible:

```bash
python visualization/multi_viewer.py
```

The window closes automatically once all games finish, or press `Q` to exit early. Final results are printed to the terminal:

```
Agent                          Score     Moves  Status
------------------------------------------------------
BeamSearch(w=10,d=15)         26,644     1,377  finished
MCTS(n=300,d=15)              18,240       934  finished
ExpectimaxSnake(d=2)           9,820       612  finished
Random                         1,364       139  finished
```

---

## Analysis

```bash
python analysis/analyze_all.py              # full evaluation + strategy plots for all agents
python analysis/evaluate_agents.py          # summary table + score distributions
python analysis/strategy.py beam_search mcts  # win/loss component breakdown for specific agents
python analysis/heuristic_trajectory.py --agent beam_search  # per-move heuristic trajectory (requires log_move_detail=True)
```

What each script produces:

- **`analyze_all.py`** — combined run: score distributions (box + violin), tile distribution stacked bar, reward component radar chart, per-agent win/loss breakdown, cross-agent strategy heatmap, score CDF, and a win-fingerprint chart showing which reward components correlate with winning for each agent.
- **`evaluate_agents.py`** — summary table (avg/median/best score, win rates, inference time) plus standalone score distribution and tile distribution plots.
- **`strategy.py`** — for each agent, compares reward component profiles between winning and losing games. Useful for understanding *why* an agent wins or loses — e.g. whether monotonicity or merge potential is the differentiating factor.
- **`heuristic_trajectory.py`** — plots how each reward component evolves move-by-move across a game. Requires games logged with `RunLogger(log_move_detail=True)` and `num_workers=1`.

Output plots are saved to `results/`.

---

## Hyperparameter Tuning

CMA-ES optimisation of reward function weights for Beam Search:

```bash
python scripts/cma_tuner.py            # fresh run
python scripts/cma_tuner.py --resume   # resume from checkpoint
```

Logs written to `logs/cma_tuning_log.jsonl`. Best weights saved to `logs/cma_best_weights.json`.

---

## Engineering Challenges

### Beam Search — designing a heuristic that actually captures game state

The core challenge with Beam Search is that it lives or dies by the quality of its evaluation function. A raw score (sum of merges) tells you nothing about whether the board is in a good position — a board with score 5000 but tiles scattered everywhere is worse than a score 3000 board with tiles organised in a snake pattern.

This required significant research into what makes a 2048 position "good". The current six-component heuristic (`tile_score`, `empty_bonus`, `monotonicity`, `corner_bonus`, `merge_potential`, `smoothness`) was arrived at iteratively, drawing from published 2048 AI approaches. Each component captures a distinct structural property:
- **Monotonicity** penalises tiles that break a decreasing gradient from the corner — the key insight that drives the snake strategy
- **Empty bonus** uses a squared count so scarcity (few empty cells) is penalised far more than abundance
- **Corner bonus** applies a harsh squared penalty when the max tile leaves the corner, making the agent extremely reluctant to let it drift

Getting the component weights right then became a hyperparameter problem, which led to adding CMA-ES tuning (`scripts/cma_tuner.py`) to search the weight space automatically.

---

### MCTS — score normalisation for stable UCB1

Standard MCTS UCB1 exploration bonus (`c * sqrt(log N / n)`) assumes reward values are roughly in [0, 1]. 2048 merge scores can reach tens of thousands, which means without normalisation the exploitation term completely dominates and MCTS degenerates to a greedy search.

The solution was running min/max normalisation across all visited nodes during the tree traversal, rescaling observed rewards to [0, 1] before applying UCB1. This stabilises exploration regardless of score magnitude and was a meaningful improvement in win rate over the unnormalised version.

---

### AlphaZero — cold start problem and compute ceiling

AlphaZero needs MCTS to generate training data (self-play), but MCTS needs a good value network to guide search — a chicken-and-egg problem. With a randomly initialised network the early self-play games are so poor that the training signal is nearly useless, and convergence is extremely slow.

**Attempted fix: pretraining on ExpectimaxSnake rollouts.** Rather than starting from random play, we generated games with `ExpectimaxSnake(d=2)` and pretrained the value/policy head on those results before beginning self-play. This gave the network a meaningful starting point and produced a faster early improvement curve compared to a cold start.

**Why it didn't fully solve the problem.** Even with pretraining, AlphaZero faces a fundamental compute constraint that the other agents don't: every training episode requires running MCTS (hundreds of simulations per move), and MCTS itself cannot be GPU-accelerated because it is sequential tree search. PPO and other neural network agents can parallelise environment rollouts and batch gradient updates across a GPU — AlphaZero cannot. The result is that reaching competitive performance requires millions of self-play episodes, each of which is expensive, making it impractical to train to convergence without significant compute.

---

### Framework design — modularity as a first-class concern

A recurring challenge when comparing search and learning agents is that they have very different interfaces: search agents need to clone the game state for lookahead, RL agents need a transition hook `(s, a, r, s', done)` for experience collection, and logging needs to capture everything without agents knowing about it.

The solution was a strict layered architecture:

- **`BaseAgent`** defines a minimal interface (`choose_action`, lifecycle hooks) that every agent implements. Swapping agents is a one-line change.
- **`InteractionModule`** is the only place that knows about all three subsystems (game, agent, evaluator). Agents never import from `framework/`; the game never imports from `agents/`.
- **`RewardFunction`** is passed into the game context at each step, so search agents can use it for lookahead evaluation without coupling to any specific implementation.

This design means the platform generalises beyond 2048: replacing `game/engine.py` with a different game engine and keeping the agent/framework layers intact is sufficient to benchmark the same algorithms on a new domain.

---

## Tests

```bash
pytest                                          # run all 107 tests
pytest --cov=game --cov=framework --cov=agents --cov-report=term-missing
```

The suite covers `game/engine.py`, `framework/evaluation.py`, `framework/logger.py`, `framework/interaction.py`, and basic agent smoke tests. All tests run in under 1 second.

---

## Configuration

`config.json` sets default game parameters:

```json
{
  "grid_size": 4,
  "tile_2_probability": 0.9,
  "initial_tiles": 2,
  "random_seed": null
}
```

Pass overrides directly to `InteractionModule(config={...})`.
