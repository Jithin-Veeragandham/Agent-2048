# 2048 AI Research Platform

A modular framework for implementing, evaluating, and comparing AI search and learning algorithms on the 2048 game. Designed as an extensible research platform where each algorithm is a self-contained agent that plugs into a common evaluation harness.

---

## Project Structure

```
2048-ai/
├── game/                   # Game engine (pure logic, no rendering)
│   ├── __init__.py
│   └── engine.py
│
├── agents/                 # AI algorithms
│   ├── __init__.py
│   ├── base.py             # BaseAgent ABC — all agents implement this
│   ├── beam_search.py      # Heuristic Beam Search
│   └── mcts.py             # Classic MCTS + Heuristic MCTS
│
├── framework/              # Experiment harness + evaluation
│   ├── __init__.py
│   ├── interaction.py      # InteractionModule — runs episodes, logs results
│   ├── evaluation.py       # RewardFunction + GameEvaluator
│   └── logger.py           # RunLogger — JSONL logging per episode
│
├── tuning/                 # Hyperparameter optimisation
│   ├── __init__.py
│   └── cma_tuner.py        # CMA-ES reward weight tuner
│
├── analysis/               # Post-hoc strategy analysis
│   ├── __init__.py
│   └── strategy.py         # Component breakdown plots, win/loss comparison
│
├── benchmarks/             # Runtime benchmarks
│   ├── __init__.py
│   ├── rollout.py          # Clone vs save/restore rollout comparison
│   └── tree_reuse.py       # Fresh tree vs tree reuse comparison
│
├── logs/                   # Run logs (JSONL, gitignored)
├── results/                # Output plots (gitignored)
├── config.json             # Default game configuration
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Run a single agent

```python
from game import Game2048
from agents import BeamSearchAgent, MCTSHeuristicAgent
from framework import InteractionModule
from framework.logger import RunLogger

agent  = BeamSearchAgent(beam_width=10, search_depth=5)
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
    logger=logger,
    num_workers=8,   # 8 games in parallel
)
module.run(num_games=100)
```

### Compare multiple agents

```python
from framework import run_agents
from agents import BeamSearchAgent, MCTSAgent

run_agents(
    config={"grid_size": 4},
    agents=[BeamSearchAgent(beam_width=10), MCTSAgent(num_simulations=300)],
    num_games=100,
    num_workers=8,
    verbose=True,
)
```

---

## Agents

### Heuristic Beam Search
Keeps the top-`k` board states (beam) at each depth level, scored by a composite heuristic.

```python
from agents import BeamSearchAgent
agent = BeamSearchAgent(beam_width=10, search_depth=5)
```

### Classic MCTS
Monte Carlo Tree Search with random rollouts and UCB1 selection.

```python
from agents import MCTSAgent
agent = MCTSAgent(num_simulations=300, rollout_depth=15)
```

### Heuristic MCTS
MCTS with greedy heuristic-guided rollouts. UCB1 uses running min/max normalisation so `c = sqrt(2)` is correct regardless of score magnitude.

```python
from agents import MCTSHeuristicAgent
agent = MCTSHeuristicAgent(num_simulations=200, rollout_depth=10)
```

### Adding a new agent
Subclass `BaseAgent` and implement `choose_action`:

```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    agent_type = "my_agent"   # used as log file prefix

    def __init__(self):
        super().__init__("MyAgent")

    def choose_action(self, state, available_moves, game_context=None):
        return available_moves[0]   # replace with your logic
```

---

## Reward Function

Six-component composite heuristic used for both evaluation and rollout guidance:

| Component | Role |
|---|---|
| `tile_score` | Weights higher tiles exponentially |
| `empty_bonus` | Rewards open + clustered empty cells |
| `monotonicity` | Rewards decreasing gradient from corner |
| `corner_bonus` | Rewards max tile in a corner |
| `merge_potential` | Rewards adjacent matching tile pairs |
| `smoothness` | Penalises large value gaps between neighbours |

```python
from framework.evaluation import RewardFunction

rf = RewardFunction(weights={
    'tile': 1.0, 'empty': 0.5, 'mono': 1.5,
    'corner': 0.0, 'merge': 0.5, 'smooth': 0.1,
})
score = rf.compute(board)
```

---

## Logging

Each `run()` call writes one JSON line per game to `logs/<agent_type>_runs.jsonl`:

```json
{
  "run_id": "uuid",
  "game_index": 0,
  "score": 34560,
  "max_tile": 2048,
  "reached_2048": true,
  "num_moves": 1243,
  "time_sec": 18.4,
  "avg_reward_breakdown": {"tile_score": 18200, "empty_bonus": 28.3, ...},
  "final_reward_breakdown": {"tile_score": 35000, ...}
}
```

---

## Strategy Analysis

Generate per-agent and cross-agent strategy plots from logs:

```bash
# Single agent
python analysis/strategy.py beam_search

# Compare multiple agents
python analysis/strategy.py beam_search mcts ppo
```

**Outputs saved to `results/`:**
- `<agent>/strategy.png` — win vs loss component bar chart
- `<agent>/correlation.png` — Pearson correlation heatmap
- `<agent>/scatter.png` — component vs final score scatter
- `comparison_strategy.png` — cross-agent win-loss delta heatmap
- `comparison_performance.png` — win rate CI, avg scores, percentiles, CDF
- `comparison_score_dist.png` — score distribution violin plots
- `comparison_win_fingerprint.png` — winning game component profiles

---

## Hyperparameter Tuning

CMA-ES optimisation of reward function weights for Beam Search:

```bash
python tuning/cma_tuner.py
python tuning/cma_tuner.py --resume   # resume from checkpoint
```

Logs written to `logs/cma_tuning_log.jsonl` and `logs/cma_best_weights.json`.

---

## Benchmarks

```bash
python benchmarks/rollout.py        # clone vs save/restore rollout speed
python benchmarks/tree_reuse.py     # fresh tree vs tree reuse speed
```

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
