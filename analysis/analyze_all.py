"""
analyze_all.py
==============

Combined analysis script: evaluation + strategy breakdown for all agents.

Generates:
  Part 1 — Evaluation (from evaluate_agents.py)
    - Summary table
    - Score distributions (box + violin)
    - Tile distribution (stacked bar)
    - Radar chart of reward components
    - Average score bar chart

  Part 2 — Strategy (from strategy.py)
    - Per-agent: win/loss component breakdown + box plots + final board state
    - Per-agent: Pearson correlation heatmap
    - Per-agent: scatter plots (each component vs score)
    - Cross-agent: win rate with 95% CI, delta heatmap, win fingerprint,
      merge-vs-tile scatter, performance summary (percentiles + CDF)

All output saved to analysis/plots/

Run:
    python analysis/analyze_all.py
"""

import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from collections import defaultdict
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────
#  CONFIG
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
        "run_id": "ea38ab60-f3b",
    },
    "DQN": {
        "file": "dqn_4x4_runs.jsonl",
        "color": "#3498db",
        "short": "DQN",
        "run_id": "d6f6f6a5-963",
    },
}

REWARD_COMPONENTS = ["tile_score", "empty_bonus", "monotonicity", "corner_bonus", "merge_potential", "smoothness"]
REWARD_LABELS     = ["Tile Score", "Empty Bonus", "Monotonicity", "Corner Bonus", "Merge Potential", "Smoothness"]
FLIP_SIGN         = {"smoothness": True, "monotonicity": False}

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_agent_data(filepath, run_id=None):
    games = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    if not games:
        return games

    runs = defaultdict(list)
    for g in games:
        runs[g.get("run_id", "unknown")].append(g)

    if run_id:
        selected = runs.get(run_id, [])
        print(f"    run_id={run_id} ({len(selected)} games) out of {len(runs)} runs")
        return selected

    latest = max(runs, key=lambda rid: runs[rid][0].get("timestamp", ""))
    latest_games = runs[latest]
    print(f"    latest run={latest} ({len(latest_games)} games) out of {len(runs)} runs")
    return latest_games


def build_stats(games):
    scores    = [g["score"] for g in games]
    tiles     = [g["highest_tile"] for g in games]
    moves     = [g["moves"] for g in games]
    wins_2048 = sum(1 for g in games if g.get("reached_2048", False))
    wins_4096 = sum(1 for g in games if g.get("reached_4096", False))
    inf_times = [g.get("avg_inference_ms", 0) for g in games]

    reward_avgs = defaultdict(list)
    for g in games:
        rb = g.get("avg_reward_breakdown", {})
        for k, v in rb.items():
            reward_avgs[k].append(v)
    reward_means = {k: np.mean(v) for k, v in reward_avgs.items()}

    return {
        "n":              len(games),
        "scores":         scores,
        "tiles":          tiles,
        "moves":          moves,
        "avg_score":      np.mean(scores),
        "med_score":      np.median(scores),
        "std_score":      np.std(scores),
        "max_score":      np.max(scores),
        "win_rate_2048":  wins_2048 / len(games) * 100,
        "win_rate_4096":  wins_4096 / len(games) * 100,
        "avg_moves":      np.mean(moves),
        "avg_inf_ms":     np.mean(inf_times),
        "reward_means":   reward_means,
    }


# ─────────────────────────────────────────────
#  LOAD ALL AGENTS
# ─────────────────────────────────────────────

data  = {}
stats = {}

print("Loading agents...")
for name, cfg in AGENTS.items():
    path = os.path.join(LOG_DIR, cfg["file"])
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: {path} not found")
        continue
    games = load_agent_data(path, run_id=cfg.get("run_id"))
    if not games:
        print(f"  [SKIP] {name}: empty")
        continue
    data[name]  = games
    stats[name] = build_stats(games)
    print(f"  [OK]   {name}: {len(games)} games, "
          f"avg={stats[name]['avg_score']:.0f}, "
          f"win2048={stats[name]['win_rate_2048']:.1f}%")

agent_names  = [n for n in AGENTS if n in stats]
agent_colors = [AGENTS[n]["color"] for n in agent_names]
short_labels = [AGENTS[n]["short"] for n in agent_names]


# ─────────────────────────────────────────────
#  PART 1 — EVALUATION
# ─────────────────────────────────────────────

print("\n" + "=" * 95)
print(f"{'Agent':<28} {'N':>4} {'Avg Score':>10} {'Med Score':>10} "
      f"{'Best':>8} {'Win%2048':>9} {'Win%4096':>9} {'Inf ms':>8}")
print("=" * 95)
for name in agent_names:
    s = stats[name]
    print(f"{name:<28} {s['n']:>4} {s['avg_score']:>10.0f} {s['med_score']:>10.0f} "
          f"{s['max_score']:>8.0f} {s['win_rate_2048']:>8.1f}% "
          f"{s['win_rate_4096']:>8.1f}% {s['avg_inf_ms']:>8.1f}")
print("=" * 95)


# 1a. Score distributions (box + violin)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Score Distributions by Agent", fontsize=14, fontweight="bold")

agent_scores = [stats[n]["scores"] for n in agent_names]

bp = axes[0].boxplot(agent_scores, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], agent_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[0].set_xticks(range(1, len(agent_names) + 1))
axes[0].set_xticklabels(short_labels, fontsize=8)
axes[0].set_ylabel("Score"); axes[0].set_title("Box Plot")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

vp = axes[1].violinplot(agent_scores, showmedians=True, showextrema=True)
for pc, color in zip(vp["bodies"], agent_colors):
    pc.set_facecolor(color); pc.set_alpha(0.7)
axes[1].set_xticks(range(1, len(agent_names) + 1))
axes[1].set_xticklabels(short_labels, fontsize=8)
axes[1].set_ylabel("Score"); axes[1].set_title("Violin Plot")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: score_distributions.png")


# 1b. Tile distribution (stacked bar — grouped buckets)
# Buckets: <512, 512, 1024, 2048, 4096, >4096
TILE_BUCKETS = [
    ("<512",  lambda t: t < 512,        "#bdc3c7"),
    ("512",   lambda t: t == 512,       "#7f8c8d"),
    ("1024",  lambda t: t == 1024,      "#f39c12"),
    ("2048",  lambda t: t == 2048,      "#e74c3c"),
    ("4096",  lambda t: t == 4096,      "#8e44ad"),
    (">4096", lambda t: t > 4096,       "#2c3e50"),
]

fig, ax = plt.subplots(figsize=(14, 6))
x      = np.arange(len(agent_names))
bottom = np.zeros(len(agent_names))

for label, predicate, color in TILE_BUCKETS:
    rates = [sum(1 for t in stats[n]["tiles"] if predicate(t)) / stats[n]["n"] * 100
             for n in agent_names]
    ax.bar(x, rates, 0.6, bottom=bottom, color=color,
           label=label, alpha=0.9, edgecolor="white", linewidth=0.5)
    bottom += np.array(rates)

ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("% of Games"); ax.set_ylim(0, 105)
ax.set_title("Max Tile Distribution per Agent", fontsize=13, fontweight="bold")
ax.legend(title="Max Tile", loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tile_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: tile_distributions.png")


# 1c. Radar chart — all 6 reward components
RADAR_COMPONENTS = ["tile_score", "empty_bonus", "monotonicity", "corner_bonus", "merge_potential", "smoothness"]
RADAR_LABELS     = ["Tile Score", "Empty Bonus", "Monotonicity", "Corner Bonus", "Merge Potential", "Smoothness"]
N_AXES = len(RADAR_COMPONENTS)
angles = [n / float(N_AXES) * 2 * math.pi for n in range(N_AXES)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
def _radar_val(means, c):
    v = means.get(c, None)
    if v is None:
        return np.nan
    return -v if FLIP_SIGN.get(c) else v

comp_values = {name: [_radar_val(stats[name]["reward_means"], c) for c in RADAR_COMPONENTS]
               for name in agent_names}
comp_array = np.array([comp_values[n] for n in agent_names], dtype=float)
comp_min   = np.nanmin(comp_array, axis=0)
comp_max   = np.nanmax(comp_array, axis=0)
comp_range = np.where(comp_max - comp_min > 0, comp_max - comp_min, 1)

radar_styles = [
    ("-",               "o"),
    ("--",              "s"),
    ("-.",              "^"),
    (":",               "D"),
    ((0, (3, 1, 1, 1)), "P"),
]
for (name, (ls, marker)) in zip(agent_names, radar_styles):
    raw = np.array(comp_values[name], dtype=float)
    normed = np.clip((raw - comp_min) / comp_range, 0, 1).tolist()
    values = normed + normed[:1]
    color  = AGENTS[name]["color"]
    ax.plot(angles, values, linestyle=ls, marker=marker, linewidth=2.2,
            markersize=7, color=color,
            label=AGENTS[name]["short"].replace("\n", " "))
    ax.fill(angles, values, alpha=0.06, color=color)

ax.set_xticks(angles[:-1]); ax.set_xticklabels(RADAR_LABELS, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
ax.set_title("Reward Component Distribution\n(normalized per component)",
             fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "radar_reward_components.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: radar_reward_components.png")


# 1d. Average score bar chart
fig, ax = plt.subplots(figsize=(12, 6))
avg_scores = [stats[n]["avg_score"] for n in agent_names]
std_scores = [stats[n]["std_score"]  for n in agent_names]
win_rates  = [stats[n]["win_rate_2048"] for n in agent_names]

bars = ax.bar(x, avg_scores, width=0.6, color=agent_colors, alpha=0.85,
              yerr=std_scores, capsize=4, error_kw=dict(ecolor="black", lw=1.5))
for bar, wr, std in zip(bars, win_rates, std_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 500,
            f"{wr:.0f}%\n2048",
            ha="center", va="bottom", fontsize=8, color="black")

ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Average Score (± std)")
ax.set_title("Average Score by Agent  (annotations show 2048 win rate)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_scores.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: avg_scores.png")


# ─────────────────────────────────────────────
#  PART 2 — STRATEGY ANALYSIS (helpers)
# ─────────────────────────────────────────────

def split_wins_losses(games):
    wins   = [g for g in games if g.get("reached_2048") or g.get("won")]
    losses = [g for g in games if not (g.get("reached_2048") or g.get("won"))]
    return wins, losses


def extract_components(games, key="avg_reward_breakdown"):
    result = {c: [] for c in REWARD_COMPONENTS}
    for g in games:
        bd = g.get(key, {})
        for c in REWARD_COMPONENTS:
            if c in bd:
                val = bd[c]
                if FLIP_SIGN.get(c):
                    val = -val
                result[c].append(val)
    return result


def normalize_within(wins_data, losses_data):
    nw, nl = {}, {}
    for c in REWARD_COMPONENTS:
        all_vals = wins_data[c] + losses_data[c]
        if not all_vals:
            nw[c] = []; nl[c] = []; continue
        lo, hi = min(all_vals), max(all_vals)
        rng = hi - lo if hi != lo else 1.0
        nw[c] = [(v - lo) / rng for v in wins_data[c]]
        nl[c] = [(v - lo) / rng for v in losses_data[c]]
    return nw, nl


def ttest_sig(a, b):
    if len(a) < 2 or len(b) < 2:
        return ""
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def wilson_ci(w, n, z=1.96):
    if n == 0: return 0, 0
    p = w / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


# ─────────────────────────────────────────────
#  2a. Per-agent strategy + correlation + scatter
# ─────────────────────────────────────────────

for name in agent_names:
    games = data[name]
    wins, losses = split_wins_losses(games)
    color = AGENTS[name]["color"]
    agent_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(agent_dir, exist_ok=True)

    print(f"\n[Strategy] {name}: {len(games)} games | "
          f"{len(wins)} wins | {len(losses)} losses | "
          f"win rate {len(wins)/len(games)*100:.1f}%")

    wins_avg   = extract_components(wins,   "avg_reward_breakdown")
    losses_avg = extract_components(losses, "avg_reward_breakdown")
    wins_final = extract_components(wins,   "final_reward_breakdown")
    losses_fin = extract_components(losses, "final_reward_breakdown")
    nw, nl     = normalize_within(wins_avg, losses_avg)

    win_means  = [np.mean(nw[c]) if nw[c] else 0 for c in REWARD_COMPONENTS]
    loss_means = [np.mean(nl[c]) if nl[c] else 0 for c in REWARD_COMPONENTS]
    win_stds   = [np.std(nw[c])  if nw[c] else 0 for c in REWARD_COMPONENTS]
    loss_stds  = [np.std(nl[c])  if nl[c] else 0 for c in REWARD_COMPONENTS]
    xc         = np.arange(len(REWARD_COMPONENTS))
    bw         = 0.35

    # ── Strategy panel (3 rows) ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Strategy Analysis: {name}", fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(3, 1, hspace=0.6)
    ax_main  = fig.add_subplot(gs[0])
    ax_dist  = fig.add_subplot(gs[1])
    ax_final = fig.add_subplot(gs[2])

    # Row 1: Win vs loss normalized bar
    ax_main.bar(xc - bw/2, win_means,  bw, label=f"Win (n={len(wins)})",
                color="#2ecc71", alpha=0.85, yerr=win_stds,  capsize=4)
    ax_main.bar(xc + bw/2, loss_means, bw, label=f"Loss (n={len(losses)})",
                color="#e74c3c", alpha=0.85, yerr=loss_stds, capsize=4)
    ax_main.set_xticks(xc)
    ax_main.set_xticklabels(REWARD_LABELS, rotation=20, ha="right", fontsize=9)
    ax_main.set_ylabel("Normalized Value (0-1)")
    ax_main.set_title(f"{name} — Avg Reward Components: Wins vs Losses",
                      fontsize=11, fontweight="bold")
    ax_main.legend(fontsize=9); ax_main.set_ylim(0, 1.45); ax_main.grid(axis="y", alpha=0.3)
    for i, (wm, lm) in enumerate(zip(win_means, loss_means)):
        delta = wm - lm
        sig   = ttest_sig(nw[REWARD_COMPONENTS[i]], nl[REWARD_COMPONENTS[i]])
        c_    = "#27ae60" if delta > 0 else "#c0392b"
        top   = max(wm, lm) + max(win_stds[i], loss_stds[i]) + 0.04
        ax_main.text(i, top,      f"{delta:+.2f}", ha="center", va="bottom",
                     fontsize=7.5, color=c_, fontweight="bold")
        ax_main.text(i, top+0.08, sig,             ha="center", va="bottom",
                     fontsize=7,   color="#555")

    # Row 2: Box plots per component
    ax_dist.set_visible(False)
    gs2      = ax_dist.get_subplotspec().subgridspec(1, len(REWARD_COMPONENTS))
    dist_axs = [fig.add_subplot(gs2[0, i]) for i in range(len(REWARD_COMPONENTS))]
    for i, (c, label) in enumerate(zip(REWARD_COMPONENTS, REWARD_LABELS)):
        axi = dist_axs[i]
        to_plot, colors_bp, pos = [], [], []
        if wins_avg[c]:   to_plot.append(wins_avg[c]);   colors_bp.append("#2ecc71"); pos.append(1)
        if losses_avg[c]: to_plot.append(losses_avg[c]); colors_bp.append("#e74c3c"); pos.append(1.6)
        if to_plot:
            bp2 = axi.boxplot(to_plot, positions=pos, patch_artist=True,
                              widths=0.4, showfliers=False,
                              medianprops={"color": "black", "linewidth": 1.5})
            for patch, col in zip(bp2["boxes"], colors_bp):
                patch.set_facecolor(col); patch.set_alpha(0.75)
        axi.set_title(label, fontsize=8, fontweight="bold")
        axi.set_xticks([]); axi.grid(axis="y", alpha=0.3)
        if i == 0: axi.set_ylabel("Raw Value", fontsize=8)
        else:      axi.tick_params(labelleft=False)
    dist_axs[0].get_figure().text(
        0.5, ax_dist.get_position().y1 + 0.01,
        "Distribution per Component (green=win, red=loss)",
        ha="center", fontsize=10)

    # Row 3: Final board breakdown
    ax_final.set_visible(False)
    gs3       = ax_final.get_subplotspec().subgridspec(1, len(REWARD_COMPONENTS))
    final_axs = [fig.add_subplot(gs3[0, i]) for i in range(len(REWARD_COMPONENTS))]
    for i, (c, label) in enumerate(zip(REWARD_COMPONENTS, REWARD_LABELS)):
        axi = final_axs[i]
        wm_ = np.mean(wins_final[c])   if wins_final[c]   else 0
        lm_ = np.mean(losses_fin[c])   if losses_fin[c]   else 0
        axi.bar([0],   [wm_], color="#2ecc71", alpha=0.85, width=0.4)
        axi.bar([0.5], [lm_], color="#e74c3c", alpha=0.85, width=0.4)
        axi.set_title(label, fontsize=8, fontweight="bold")
        axi.set_xticks([0, 0.5]); axi.set_xticklabels(["W", "L"], fontsize=8)
        axi.grid(axis="y", alpha=0.3)
        if i == 0: axi.set_ylabel("Raw Value", fontsize=8)
        else:      axi.tick_params(labelleft=False)
    final_axs[0].get_figure().text(
        0.5, ax_final.get_position().y1 + 0.01,
        "Final Board State Breakdown", ha="center", fontsize=10)

    plt.savefig(os.path.join(agent_dir, "strategy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}/strategy.png")

    # Print key deltas
    deltas = [(REWARD_LABELS[i], win_means[i] - loss_means[i])
              for i in range(len(REWARD_COMPONENTS))]
    deltas.sort(key=lambda t: abs(t[1]), reverse=True)
    print(f"  Key win-loss differences (normalized):")
    for label, d in deltas:
        direction = "higher in wins" if d > 0 else "higher in losses"
        print(f"    {label:<18} {d:+.3f}  ({direction})")

    # ── Correlation heatmap ────────────────────────────────────────────
    scores_arr = np.array([g["score"] for g in games])
    won_arr    = np.array([1 if g.get("reached_2048") or g.get("won") else 0 for g in games])
    comp_vals  = {}
    for c in REWARD_COMPONENTS:
        vals = []
        for g in games:
            val = g.get("avg_reward_breakdown", {}).get(c, 0)
            if FLIP_SIGN.get(c): val = -val
            vals.append(val)
        comp_vals[c] = np.array(vals)

    targets     = {"Final Score": scores_arr, "Win (0/1)": won_arr}
    corr_matrix = np.zeros((len(REWARD_COMPONENTS), len(targets)))
    for i, c in enumerate(REWARD_COMPONENTS):
        for j, tvals in enumerate(targets.values()):
            if np.std(comp_vals[c]) > 0:
                corr_matrix[i, j] = np.corrcoef(comp_vals[c], tvals)[0, 1]

    fig1, ax1 = plt.subplots(figsize=(5, 6))
    im = ax1.imshow(corr_matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(targets))); ax1.set_xticklabels(list(targets.keys()), fontsize=10)
    ax1.set_yticks(range(len(REWARD_COMPONENTS))); ax1.set_yticklabels(REWARD_LABELS, fontsize=10)
    ax1.set_title(f"{name}\nPearson Correlation with Outcome", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax1, label="Correlation")
    for i in range(len(REWARD_COMPONENTS)):
        for j in range(len(targets)):
            ax1.text(j, i, f"{corr_matrix[i, j]:+.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(agent_dir, "correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {name}/correlation.png")

    # ── Scatter plots ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    fig2.suptitle(f"{name} — Component vs Final Score", fontsize=13, fontweight="bold")
    axes2 = axes2.flatten()
    scatter_colors = ["#2ecc71" if w else "#e74c3c" for w in won_arr]

    for i, (c, label) in enumerate(zip(REWARD_COMPONENTS, REWARD_LABELS)):
        ax_ = axes2[i]
        ax_.scatter(comp_vals[c], scores_arr, c=scatter_colors, alpha=0.65, s=40, edgecolors="none")
        if np.std(comp_vals[c]) > 0:
            z   = np.polyfit(comp_vals[c], scores_arr, 1)
            xs  = np.linspace(comp_vals[c].min(), comp_vals[c].max(), 100)
            ax_.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1.2, alpha=0.6)
            corr = np.corrcoef(comp_vals[c], scores_arr)[0, 1]
            ax_.set_title(f"{label}  (r={corr:+.2f})", fontsize=10, fontweight="bold")
        else:
            ax_.set_title(label, fontsize=10)
        ax_.set_xlabel("Avg Component Value", fontsize=8)
        ax_.set_ylabel("Final Score", fontsize=8)
        ax_.grid(alpha=0.3)
        if i == 0:
            ax_.legend(handles=[Patch(color="#2ecc71", label="Win"),
                                 Patch(color="#e74c3c", label="Loss")], fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(agent_dir, "scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {name}/scatter.png")


# ─────────────────────────────────────────────
#  2b. Cross-agent comparison
# ─────────────────────────────────────────────

print("\n[Cross-agent comparison]")

all_wins_map   = {n: split_wins_losses(data[n])[0] for n in agent_names}
all_losses_map = {n: split_wins_losses(data[n])[1] for n in agent_names}
colors_list    = agent_colors
agent_labels   = [n.replace("-", "\n") for n in agent_names]

# ── Figure C1: Win rate + delta heatmap ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cross-Agent Strategy Comparison", fontsize=13, fontweight="bold")

win_rates_cross = [len(all_wins_map[n]) / stats[n]["n"] * 100 for n in agent_names]
delta_matrix    = {c: [] for c in REWARD_COMPONENTS}

for name in agent_names:
    wa = extract_components(all_wins_map[name],   "avg_reward_breakdown")
    la = extract_components(all_losses_map[name], "avg_reward_breakdown")
    nw_, nl_ = normalize_within(wa, la)
    for c in REWARD_COMPONENTS:
        wm_ = np.mean(nw_[c]) if nw_[c] else 0
        lm_ = np.mean(nl_[c]) if nl_[c] else 0
        delta_matrix[c].append(wm_ - lm_)

axes[0].bar(agent_labels, win_rates_cross, color=colors_list, alpha=0.85)
axes[0].set_ylabel("Win Rate (%)"); axes[0].set_title("Win Rate by Agent")
axes[0].set_ylim(0, 110); axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(win_rates_cross):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

mat = np.array([delta_matrix[c] for c in REWARD_COMPONENTS])
im  = axes[1].imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
axes[1].set_xticks(range(len(agent_names))); axes[1].set_xticklabels(agent_labels, fontsize=9)
axes[1].set_yticks(range(len(REWARD_COMPONENTS))); axes[1].set_yticklabels(REWARD_LABELS, fontsize=9)
axes[1].set_title("Win-Loss Component Delta\n(green = component higher in wins)")
plt.colorbar(im, ax=axes[1], label="Normalized delta")
for i in range(len(REWARD_COMPONENTS)):
    for j in range(len(agent_names)):
        axes[1].text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_strategy.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: comparison_strategy.png")

# ── Figure C2: Score distribution all + wins only ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Score Distribution by Agent", fontsize=13, fontweight="bold")

for ax, label, fn in [
    (axes[0], "All Games", lambda n: data[n]),
    (axes[1], "Wins Only", lambda n: all_wins_map[n]),
]:
    subsets = [fn(n) for n in agent_names]
    scores_sub = [[g["score"] for g in s] for s in subsets]
    valid = [(i, s) for i, s in enumerate(scores_sub) if len(s) > 1]
    if not valid:
        ax.set_title(label); continue
    valid_idx, valid_scores = zip(*valid)
    parts = ax.violinplot(list(valid_scores), positions=list(valid_idx),
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [colors_list[i] for i in valid_idx]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(1.2)
    ax.set_xticks(range(len(agent_names))); ax.set_xticklabels(agent_labels, fontsize=9)
    ax.set_ylabel("Final Score"); ax.set_title(label); ax.grid(axis="y", alpha=0.3)
    for i, s in zip(valid_idx, valid_scores):
        ax.text(i, max(s) * 1.02, f"med={int(np.median(s))}",
                ha="center", fontsize=7.5, color="#333")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_score_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: comparison_score_dist.png")

# ── Figure C3: Win fingerprint bar chart ────────────────────────────
all_win_vals = {c: [] for c in REWARD_COMPONENTS}
for name in agent_names:
    d = extract_components(all_wins_map[name], "avg_reward_breakdown")
    for c in REWARD_COMPONENTS:
        all_win_vals[c].extend(d[c])

global_ranges = {}
for c in REWARD_COMPONENTS:
    vals = all_win_vals[c]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    global_ranges[c] = (lo, hi - lo if hi != lo else 1.0)

fig, ax = plt.subplots(figsize=(13, 5))
bw2 = 0.8 / len(agent_names)
xc2 = np.arange(len(REWARD_COMPONENTS))

for idx, (name, color) in enumerate(zip(agent_names, colors_list)):
    d = extract_components(all_wins_map[name], "avg_reward_breakdown")
    means, stds = [], []
    for c in REWARD_COMPONENTS:
        lo, rng = global_ranges[c]
        normed  = [(v - lo) / rng for v in d[c]] if d[c] else [0]
        means.append(np.mean(normed)); stds.append(np.std(normed))
    offset = (idx - (len(agent_names) - 1) / 2) * bw2
    ax.bar(xc2 + offset, means, bw2, label=name, color=color,
           alpha=0.85, yerr=stds, capsize=3, error_kw={"linewidth": 1})

ax.set_xticks(xc2); ax.set_xticklabels(REWARD_LABELS, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Normalized Value (0-1, globally scaled)")
ax.set_title("Winning Strategy Fingerprint: Component Profile in Winning Games",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim(0, 1.35); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_win_fingerprint.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: comparison_win_fingerprint.png")

# ── Figure C4: Merge potential vs tile score scatter ─────────────────
fig, axes = plt.subplots(1, len(agent_names), figsize=(7 * len(agent_names), 5), sharey=True)
if len(agent_names) == 1:
    axes = [axes]
fig.suptitle("Merge Potential vs Tile Score (green=win, red=loss)",
             fontsize=12, fontweight="bold")

for ax, name in zip(axes, agent_names):
    games   = data[name]
    won_    = np.array([1 if g.get("reached_2048") or g.get("won") else 0 for g in games])
    mp      = np.array([g.get("avg_reward_breakdown", {}).get("merge_potential", 0) for g in games])
    ts      = np.array([g.get("avg_reward_breakdown", {}).get("tile_score", 0)      for g in games])
    clrs    = ["#2ecc71" if w else "#e74c3c" for w in won_]
    ax.scatter(mp, ts, c=clrs, alpha=0.65, s=45, edgecolors="none")
    ax.set_xlabel("Avg Merge Potential", fontsize=10)
    ax.set_ylabel("Avg Tile Score",      fontsize=10)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    for mask, lbl, col in [(won_ == 1, "win", "#27ae60"), (won_ == 0, "loss", "#c0392b")]:
        if mask.sum():
            ax.scatter(mp[mask].mean(), ts[mask].mean(), marker="*", s=250,
                       color=col, edgecolors="black", linewidths=0.5, zorder=5, label=f"{lbl} centroid")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_merge_vs_tile.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: comparison_merge_vs_tile.png")

# ── Figure C5: Performance summary (CI + percentiles + CDF) ──────────
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle("Performance Summary", fontsize=13, fontweight="bold")

# Win rate with Wilson CI
ax = axes5[0, 0]
wr_vals, wr_lo, wr_hi = [], [], []
for name in agent_names:
    n = stats[name]["n"]; w = len(all_wins_map[name])
    wr_vals.append(w / n * 100)
    lo, hi = wilson_ci(w, n)
    wr_lo.append((w/n - lo) * 100)
    wr_hi.append((hi - w/n) * 100)
ax.bar(agent_labels, wr_vals, color=colors_list, alpha=0.85)
ax.errorbar(range(len(agent_names)), wr_vals,
            yerr=[wr_lo, wr_hi], fmt="none", color="black", capsize=6, linewidth=1.5)
ax.set_ylabel("Win Rate (%)"); ax.set_title("Win Rate (95% Wilson CI)")
ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)
for i, (v, hi) in enumerate(zip(wr_vals, wr_hi)):
    ax.text(i, v + hi + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

# Avg score: all / wins / losses
ax = axes5[0, 1]
bw3 = 0.25
xa  = np.arange(len(agent_names))
avg_all_  = [stats[n]["avg_score"] for n in agent_names]
avg_wins_ = [np.mean([g["score"] for g in all_wins_map[n]])   if all_wins_map[n]   else 0 for n in agent_names]
avg_loss_ = [np.mean([g["score"] for g in all_losses_map[n]]) if all_losses_map[n] else 0 for n in agent_names]
ax.bar(xa - bw3, avg_all_,  bw3, label="All",    color="#95a5a6", alpha=0.85)
ax.bar(xa,       avg_wins_, bw3, label="Wins",   color="#2ecc71", alpha=0.85)
ax.bar(xa + bw3, avg_loss_, bw3, label="Losses", color="#e74c3c", alpha=0.85)
ax.set_xticks(xa); ax.set_xticklabels(agent_labels, fontsize=9)
ax.set_ylabel("Avg Final Score"); ax.set_title("Average Score: All / Wins / Losses")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
for i, (a_, b_, c_) in enumerate(zip(avg_all_, avg_wins_, avg_loss_)):
    ax.text(i-bw3, a_*1.01, f"{int(a_):,}", ha="center", fontsize=7)
    ax.text(i,     b_*1.01, f"{int(b_):,}", ha="center", fontsize=7)
    ax.text(i+bw3, c_*1.01, f"{int(c_):,}", ha="center", fontsize=7)

# Score percentiles
ax = axes5[1, 0]
percentiles = [25, 50, 75, 90]
xp = np.arange(len(percentiles))
bw4 = 0.8 / len(agent_names)
for idx, (name, color) in enumerate(zip(agent_names, colors_list)):
    sc   = [g["score"] for g in data[name]]
    pcts = [np.percentile(sc, p) for p in percentiles]
    off  = (idx - (len(agent_names)-1)/2) * bw4
    ax.bar(xp + off, pcts, bw4, label=name, color=color, alpha=0.85)
ax.set_xticks(xp); ax.set_xticklabels([f"P{p}" for p in percentiles], fontsize=10)
ax.set_ylabel("Score"); ax.set_title("Score Percentiles (all games)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# CDF
ax = axes5[1, 1]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
for name, color, ls in zip(agent_names, colors_list, linestyles):
    sc_sorted = np.sort([g["score"] for g in data[name]])
    cdf = np.arange(1, len(sc_sorted)+1) / len(sc_sorted)
    ax.plot(sc_sorted, cdf, color=color, linewidth=2, linestyle=ls, label=name)
ax.axvline(x=20480, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="2048 threshold (~20k)")
ax.set_xlabel("Final Score"); ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF of Final Scores"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_performance.png"), dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"Saved: comparison_performance.png")

# ── Figure C6: Heuristic percentile box plots (grouped by component) ─
FLIP_SIGN = {"monotonicity": -1, "corner_bonus": -1}

fig6, axes6 = plt.subplots(2, 3, figsize=(16, 10))
fig6.suptitle("Heuristic Component Distribution per Agent\n(avg per game, normalized to [0,1] per component)",
              fontsize=13, fontweight="bold")

for ax, comp, label in zip(axes6.flat, REWARD_COMPONENTS, REWARD_LABELS):
    # Collect per-game avg values for each agent
    agent_vals = []
    for name in agent_names:
        vals = []
        for g in data[name]:
            bd = g.get("avg_reward_breakdown", {})
            if comp in bd:
                v = bd[comp] * FLIP_SIGN.get(comp, 1)
                vals.append(v)
        agent_vals.append(vals)

    # Normalize across all agents together
    all_vals = [v for av in agent_vals for v in av]
    vmin, vmax = min(all_vals), max(all_vals)
    rng = vmax - vmin if vmax != vmin else 1
    norm_vals = [[(v - vmin) / rng for v in av] for av in agent_vals]

    bp = ax.boxplot(norm_vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(agent_names) + 1))
    ax.set_xticklabels([AGENTS[n]["short"].replace("\n", " ") for n in agent_names],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_heuristic_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close(fig6)
print(f"Saved: comparison_heuristic_boxplots.png")

# ── Figure C7: Cross-agent heuristic trajectories over game progress ──
N_BINS = 100

def interpolate_trajectory(vals, n_bins=N_BINS):
    """Resample a trajectory of any length to n_bins evenly spaced points."""
    n = len(vals)
    if n < 2:
        return None
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, vals)

fig7, axes7 = plt.subplots(2, 3, figsize=(16, 10))
fig7.suptitle("Avg Rate of Change per Heuristic Over Game Progress\n(positive = gaining, negative = declining, zero = plateau)",
              fontsize=13, fontweight="bold")

x_pct = np.linspace(0, 100, N_BINS)
x_mid = (x_pct[:-1] + x_pct[1:]) / 2   # midpoints for delta plot
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
SMOOTH_W = 8  # rolling average window for deltas

def smooth(arr, w):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')

for ax, comp, label in zip(axes7.flat, REWARD_COMPONENTS, REWARD_LABELS):
    flip = FLIP_SIGN.get(comp, 1)
    all_curves = {}

    for name in agent_names:
        curves = []
        for g in data[name]:
            moves = g.get("move_reward_breakdowns")
            if not moves:
                continue
            vals = [m[comp] * flip for m in moves if comp in m]
            if len(vals) < 2:
                continue
            interp = interpolate_trajectory(vals)
            if interp is not None:
                curves.append(interp)
        if curves:
            all_curves[name] = np.array(curves)

    if not all_curves:
        ax.set_title(label + " (no move data)")
        continue

    # Normalize across all agents so deltas are comparable
    global_min = min(c.min() for c in all_curves.values())
    global_max = max(c.max() for c in all_curves.values())
    rng = global_max - global_min if global_max != global_min else 1

    for name, color, ls in zip(agent_names, colors_list, linestyles):
        if name not in all_curves:
            continue
        curves_norm = (all_curves[name] - global_min) / rng
        # Delta: diff between consecutive bins, averaged across games
        deltas = np.diff(curves_norm, axis=1)          # shape (n_games, N_BINS-1)
        mean_delta = smooth(deltas.mean(axis=0), SMOOTH_W)
        std_delta  = deltas.std(axis=0)
        ax.plot(x_mid, mean_delta, color=color, linewidth=2, linestyle=ls,
                label=AGENTS[name]["short"].replace("\n", " "))
        ax.fill_between(x_mid,
                        mean_delta - std_delta * 0.25,
                        mean_delta + std_delta * 0.25,
                        color=color, alpha=0.12)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Game progress (%)")
    ax.set_ylabel("Avg gain per step (normalized)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_heuristic_trajectories.png"), dpi=150, bbox_inches="tight")
plt.close(fig7)
print(f"Saved: comparison_heuristic_trajectories.png")


# ── Figure C8: Win-advantage heatmap (one panel per agent) ────────────
# Rows = heuristic components, Cols = 4 game phases
# Cell colour = mean_norm(wins) - mean_norm(losses) at that phase
# One presentation-ready figure; shared diverging colorbar.

PHASES_C8    = [(0, 25), (25, 50), (50, 75), (75, 100)]
PHASE_LABELS_C8 = ["Early\n0-25%", "Mid-Early\n25-50%",
                   "Mid-Late\n50-75%", "Late\n75-100%"]

def _phase_component_means(games, n_bins=100):
    """Return (n_games, n_components, n_phases) normalized values."""
    result = []
    for g in games:
        moves = g.get("move_reward_breakdowns")
        if not moves:
            continue
        game_matrix = []
        for comp in REWARD_COMPONENTS:
            flip = -1 if FLIP_SIGN.get(comp) else 1
            vals = np.array([m.get(comp, 0) * flip for m in moves], dtype=float)
            lo, hi = vals.min(), vals.max()
            normed = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
            interp = interpolate_trajectory(normed, n_bins)
            phase_vals = [interp[lo_p:hi_p].mean() for lo_p, hi_p in PHASES_C8]
            game_matrix.append(phase_vals)
        result.append(game_matrix)
    return np.array(result)  # (n_games, n_components, n_phases)

# Only agents that have move-level data
agents_with_moves = [n for n in agent_names
                     if any(g.get("move_reward_breakdowns") for g in data[n])]

if agents_with_moves:
    n_agents = len(agents_with_moves)
    fig8, axes8 = plt.subplots(1, n_agents, figsize=(5 * n_agents, 7),
                                constrained_layout=True)
    if n_agents == 1:
        axes8 = [axes8]

    fig8.suptitle(
        "Win Advantage by Heuristic Component & Game Phase\n"
        "(green = higher in wins, red = higher in losses)",
        fontsize=13, fontweight="bold",
    )

    vmax = 0.0
    heat_data = {}
    for name in agents_with_moves:
        games_w = [g for g in data[name] if g.get("reached_2048")]
        games_l = [g for g in data[name] if not g.get("reached_2048")]
        if not games_w or not games_l:
            continue
        arr_w = _phase_component_means(games_w)   # (nW, 6, 4)
        arr_l = _phase_component_means(games_l)   # (nL, 6, 4)
        if arr_w.size == 0 or arr_l.size == 0:
            continue
        diff = arr_w.mean(axis=0) - arr_l.mean(axis=0)  # (6, 4)
        heat_data[name] = diff
        vmax = max(vmax, np.abs(diff).max())

    if not heat_data:
        print("  [SKIP] Figure C8: no agents with win/loss move data")
    else:
        vmax = max(vmax, 0.05)  # floor so colorbar isn't empty
        im8 = None
        for ax, name in zip(axes8, agents_with_moves):
            if name not in heat_data:
                ax.set_visible(False)
                continue
            diff = heat_data[name]
            im8 = ax.imshow(diff, aspect="auto", cmap="RdYlGn",
                            vmin=-vmax, vmax=vmax, interpolation="nearest")

            # Annotate cells
            for r in range(len(REWARD_COMPONENTS)):
                for c in range(len(PHASES_C8)):
                    val = diff[r, c]
                    color = "black" if abs(val) < vmax * 0.6 else "white"
                    ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

            short = AGENTS[name]["short"].replace("\n", " ")
            n_w = sum(1 for g in data[name] if g.get("reached_2048"))
            n_l = sum(1 for g in data[name] if not g.get("reached_2048"))
            ax.set_title(f"{short}\n(W={n_w}, L={n_l})",
                         fontsize=11, fontweight="bold",
                         color=AGENTS[name]["color"])
            ax.set_xticks(range(len(PHASES_C8)))
            ax.set_xticklabels(PHASE_LABELS_C8, fontsize=8)
            ax.set_yticks(range(len(REWARD_COMPONENTS)))
            ax.set_yticklabels(REWARD_LABELS, fontsize=9)
            ax.set_xlabel("Game Phase", fontsize=9)

        if im8 is not None:
            cbar = fig8.colorbar(im8, ax=axes8, shrink=0.7, pad=0.02)
            cbar.set_label("Win mean − Loss mean (normalised)", fontsize=9)

        plt.savefig(os.path.join(OUTPUT_DIR, "comparison_win_advantage_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig8)
        print(f"Saved: comparison_win_advantage_heatmap.png")


print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print(f"  Per-agent subfolders: {', '.join(agent_names)}")
