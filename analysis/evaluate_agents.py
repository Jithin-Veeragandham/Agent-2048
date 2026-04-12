"""
evaluate_agents.py
==================

Evaluates all agents from existing logs and generates:
1. Summary table (avg score, win rate, best tile, inference time)
2. Score distribution plots (histogram + box plot)
3. Tile distribution bar chart
4. Radar chart of avg reward function components per agent

Run:
    python analysis/evaluate_agents.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import math
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIG: which logs to include
# ─────────────────────────────────────────────

LOG_DIR = "logs"

AGENTS = {
    "ExpectimaxSnake": {
        "file": "expectimax_snake_runs.jsonl",
        "color": "#2ecc71",
        "short": "Expectimax\nSnake",
        "run_id": "20746c9b-5ff",
    },
    "MCTS": {
        "file": "mcts_runs.jsonl",
        "color": "#9b59b6",
        "short": "MCTS",
        "run_id": "17602ce1-5bc",
    },
    "PPO": {
        "file": "PPO_runs.jsonl",
        "color": "#e74c3c",
        "short": "PPO",
        "run_id": "aee9fc06-16d",
    },
    "BeamSearch": {
        "file": "beam_search_runs.jsonl",
        "color": "#e67e22",
        "short": "Beam\nSearch",
        "run_id": "d2b82f14-d47",
    },
    "NTuple": {
        "file": "ntuple_agent_runs.jsonl",
        "color": "#1abc9c",
        "short": "NTuple",
        "run_id": "eeb816ca-32d",
    }, 
    #"DQN": {
   #     "file": "dqn_4x4_runs.jsonl",
    #    "color": "#3498db",
     #   "short": "DQN",
      #  "run_id": "d6f6f6a5-963",
    #},
}

REWARD_COMPONENTS = [
    "tile_score",
    "empty_bonus",
    "monotonicity",
    "merge_potential",
    "smoothness",
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_agent_data(filepath, run_id=None):
    """Load game records from a jsonl file.

    If run_id is specified, load that specific run.
    Otherwise load the latest run by timestamp.
    """
    games = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))

    if not games:
        return games

    from collections import defaultdict
    runs = defaultdict(list)
    for g in games:
        runs[g.get("run_id", "unknown")].append(g)

    if run_id:
        selected_games = runs.get(run_id, [])
        print(f"    Using run {run_id} ({len(selected_games)} games) "
              f"out of {len(runs)} runs ({len(games)} total)")
        return selected_games

    # Default: latest run by timestamp
    latest_run_id = max(runs, key=lambda rid: runs[rid][0].get("timestamp", ""))
    latest_games  = runs[latest_run_id]
    print(f"    Using run {latest_run_id} ({len(latest_games)} games) "
          f"out of {len(runs)} runs ({len(games)} total)")
    return latest_games


def build_stats(games):
    """Compute summary statistics from game records."""
    scores      = [g["score"] for g in games]
    tiles       = [g["highest_tile"] for g in games]
    moves       = [g["moves"] for g in games]
    wins_2048   = sum(1 for g in games if g.get("reached_2048", False))
    wins_4096   = sum(1 for g in games if g.get("reached_4096", False))
    inf_times   = [g.get("avg_inference_ms", 0) for g in games]

    # Reward breakdown averages
    reward_avgs = defaultdict(list)
    for g in games:
        rb = g.get("avg_reward_breakdown", {})
        for k, v in rb.items():
            reward_avgs[k].append(v)
    reward_means = {k: np.mean(v) for k, v in reward_avgs.items()}

    return {
        "n":            len(games),
        "scores":       scores,
        "tiles":        tiles,
        "moves":        moves,
        "avg_score":    np.mean(scores),
        "med_score":    np.median(scores),
        "std_score":    np.std(scores),
        "max_score":    np.max(scores),
        "win_rate_2048": wins_2048 / len(games) * 100,
        "win_rate_4096": wins_4096 / len(games) * 100,
        "avg_moves":    np.mean(moves),
        "avg_inf_ms":   np.mean(inf_times),
        "reward_means": reward_means,
    }


# ─────────────────────────────────────────────
#  LOAD ALL
# ─────────────────────────────────────────────

data = {}
stats = {}

for name, cfg in AGENTS.items():
    path = os.path.join(LOG_DIR, cfg["file"])
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: {path} not found")
        continue
    games = load_agent_data(path, run_id=cfg.get("run_id"))
    if not games:
        print(f"  [SKIP] {name}: empty file")
        continue
    data[name]  = games
    stats[name] = build_stats(games)
    print(f"  [OK]   {name}: {len(games)} games, "
          f"avg={stats[name]['avg_score']:.0f}, "
          f"win2048={stats[name]['win_rate_2048']:.1f}%")


# ─────────────────────────────────────────────
#  1. SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + "=" * 90)
print(f"{'Agent':<28} {'N':>4} {'Avg Score':>10} {'Med Score':>10} "
      f"{'Best':>8} {'Win%2048':>9} {'Win%4096':>9} {'Inf ms':>8}")
print("=" * 90)

for name in AGENTS:
    if name not in stats:
        continue
    s = stats[name]
    print(f"{name:<28} {s['n']:>4} {s['avg_score']:>10.0f} {s['med_score']:>10.0f} "
          f"{s['max_score']:>8.0f} {s['win_rate_2048']:>8.1f}% "
          f"{s['win_rate_4096']:>8.1f}% {s['avg_inf_ms']:>8.1f}")
print("=" * 90)


# ─────────────────────────────────────────────
#  2. SCORE DISTRIBUTIONS (box + violin)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Score Distributions by Agent", fontsize=14, fontweight="bold")

agent_names   = [n for n in AGENTS if n in stats]
agent_scores  = [stats[n]["scores"] for n in agent_names]
agent_colors  = [AGENTS[n]["color"] for n in agent_names]
short_labels  = [AGENTS[n]["short"] for n in agent_names]

# Box plot
bp = axes[0].boxplot(agent_scores, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], agent_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticks(range(1, len(agent_names) + 1))
axes[0].set_xticklabels(short_labels, fontsize=8)
axes[0].set_ylabel("Score")
axes[0].set_title("Box Plot")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Violin plot
vp = axes[1].violinplot(agent_scores, showmedians=True, showextrema=True)
for pc, color in zip(vp["bodies"], agent_colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
axes[1].set_xticks(range(1, len(agent_names) + 1))
axes[1].set_xticklabels(short_labels, fontsize=8)
axes[1].set_ylabel("Score")
axes[1].set_title("Violin Plot")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/score_distributions.png")


# ─────────────────────────────────────────────
#  3. TILE DISTRIBUTION (stacked bar — max tile reached per game)
# ─────────────────────────────────────────────

TILE_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
TILE_COLORS  = ["#ecf0f1", "#bdc3c7", "#95a5a6", "#7f8c8d", "#f39c12", "#e74c3c", "#8e44ad"]

fig, ax = plt.subplots(figsize=(14, 6))

x     = np.arange(len(agent_names))
width = 0.6
bottom = np.zeros(len(agent_names))

for tile, color in zip(TILE_BUCKETS, TILE_COLORS):
    rates = []
    for name in agent_names:
        n     = stats[name]["n"]
        # % of games where max tile == this bucket
        count = sum(1 for t in stats[name]["tiles"] if t == tile)
        rates.append(count / n * 100)
    bars = ax.bar(x, rates, width, bottom=bottom, color=color,
                  label=str(tile), alpha=0.9, edgecolor="white", linewidth=0.5)
    bottom += np.array(rates)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("% of Games")
ax.set_ylim(0, 105)
ax.set_title("Max Tile Distribution per Agent", fontsize=13, fontweight="bold")
ax.legend(title="Max Tile", loc="upper right", fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tile_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/tile_distributions.png")


# ─────────────────────────────────────────────
#  4. RADAR CHART — reward component breakdown
# ─────────────────────────────────────────────

RADAR_COMPONENTS = ["tile_score", "empty_bonus", "merge_potential", "smoothness"]
RADAR_LABELS     = ["Tile Score", "Empty Bonus", "Merge Potential", "Smoothness"]

# Monotonicity is usually negative — handle separately
# Only use components that are positive or normalize properly

N_AXES = len(RADAR_COMPONENTS)
angles = [n / float(N_AXES) * 2 * math.pi for n in range(N_AXES)]
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# Normalize each component to [0, 1] across all agents
comp_values = {}
for name in agent_names:
    comp_values[name] = [
        stats[name]["reward_means"].get(c, 0) for c in RADAR_COMPONENTS
    ]

# Get min/max per component for normalization
comp_array = np.array([comp_values[n] for n in agent_names])
comp_min   = comp_array.min(axis=0)
comp_max   = comp_array.max(axis=0)
comp_range = np.where(comp_max - comp_min > 0, comp_max - comp_min, 1)

for name in agent_names:
    raw    = np.array(comp_values[name])
    normed = (raw - comp_min) / comp_range
    values = normed.tolist()
    values += values[:1]  # close

    color = AGENTS[name]["color"]
    ax.plot(angles, values, "o-", linewidth=2, color=color, label=AGENTS[name]["short"].replace("\n", " "))
    ax.fill(angles, values, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
ax.set_title("Reward Component Distribution\n(normalized per component)",
             fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "radar_reward_components.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/radar_reward_components.png")


# ─────────────────────────────────────────────
#  5. AVG SCORE BAR CHART (clean summary)
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

avg_scores  = [stats[n]["avg_score"] for n in agent_names]
std_scores  = [stats[n]["std_score"] for n in agent_names]
win_rates   = [stats[n]["win_rate_2048"] for n in agent_names]

bars = ax.bar(x, avg_scores, width=0.6,
              color=agent_colors, alpha=0.85,
              yerr=std_scores, capsize=4, error_kw=dict(ecolor="black", lw=1.5))

# Annotate win rate on top of each bar
for i, (bar, wr) in enumerate(zip(bars, win_rates)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std_scores[i] + 500,
            f"{wr:.0f}%\n2048",
            ha="center", va="bottom", fontsize=8, color="black")

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Average Score (± std)")
ax.set_title("Average Score by Agent  (annotations show 2048 win rate)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_scores.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/avg_scores.png")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
