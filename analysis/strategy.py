"""
analyze_strategy.py
===================
Analyzes reward function component breakdowns from agent log files
to understand what strategies each agent uses to win or lose.

Usage::

    python analyze_strategy.py                          # defaults to all log files
    python analyze_strategy.py beam_search              # specific agent
    python analyze_strategy.py beam_search mcts         # compare two agents
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display — save only
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import List, Dict

LOGS_DIR = "logs"
COMPONENTS = ["tile_score", "empty_bonus", "monotonicity", "corner_bonus", "merge_potential", "smoothness"]
COMPONENT_LABELS = ["Tile Score", "Empty Bonus", "Monotonicity", "Corner Bonus", "Merge Potential", "Smoothness"]

# Smoothness is a penalty — flip sign so "higher = better" for all
FLIP_SIGN = {"smoothness": True, "monotonicity": False}

AGENT_COLORS = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c']


def load_runs(agent_type: str) -> List[Dict]:
    path = os.path.join(LOGS_DIR, f"{agent_type}_runs.jsonl")
    if not os.path.exists(path):
        print(f"  [!] No log file found: {path}")
        return []
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    if runs:
        latest_run_id = max(runs, key=lambda r: r.get('timestamp', ''))['run_id']
        runs = [r for r in runs if r['run_id'] == latest_run_id]
        print(f"  Using latest run_id: {latest_run_id} ({len(runs)} games)")

    return runs


def split_wins_losses(runs: List[Dict]):
    wins   = [r for r in runs if r.get('reached_2048') or r.get('won')]
    losses = [r for r in runs if not (r.get('reached_2048') or r.get('won'))]
    return wins, losses


def extract_component_values(runs: List[Dict], key: str = "avg_reward_breakdown") -> Dict[str, List[float]]:
    data = {c: [] for c in COMPONENTS}
    for r in runs:
        bd = r.get(key, {})
        for c in COMPONENTS:
            if c in bd:
                val = bd[c]
                if FLIP_SIGN.get(c):
                    val = -val
                data[c].append(val)
    return data


def normalize_across_all(wins_data, losses_data):
    normed_wins, normed_losses = {}, {}
    for c in COMPONENTS:
        all_vals = wins_data[c] + losses_data[c]
        if not all_vals:
            normed_wins[c] = []
            normed_losses[c] = []
            continue
        lo, hi = min(all_vals), max(all_vals)
        rng = hi - lo if hi != lo else 1.0
        normed_wins[c]   = [(v - lo) / rng for v in wins_data[c]]
        normed_losses[c] = [(v - lo) / rng for v in losses_data[c]]
    return normed_wins, normed_losses


def ttest_sig(a, b):
    """Returns significance string: *** p<0.001, ** p<0.01, * p<0.05, ns"""
    if len(a) < 2 or len(b) < 2:
        return ''
    _, p = stats.ttest_ind(a, b, equal_var=False)
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def analyze_agent(agent_type: str, ax_main, ax_dist, ax_final):
    runs = load_runs(agent_type)
    if not runs:
        return

    wins, losses = split_wins_losses(runs)
    print(f"\n{agent_type}: {len(runs)} games  |  {len(wins)} wins  |  {len(losses)} losses  |  win rate: {len(wins)/len(runs)*100:.1f}%")

    wins_avg     = extract_component_values(wins,   "avg_reward_breakdown")
    losses_avg   = extract_component_values(losses, "avg_reward_breakdown")
    wins_final   = extract_component_values(wins,   "final_reward_breakdown")
    losses_final = extract_component_values(losses, "final_reward_breakdown")

    norm_wins, norm_losses = normalize_across_all(wins_avg, losses_avg)

    x     = np.arange(len(COMPONENTS))
    width = 0.35

    win_means  = [np.mean(norm_wins[c])   if norm_wins[c]   else 0 for c in COMPONENTS]
    loss_means = [np.mean(norm_losses[c]) if norm_losses[c] else 0 for c in COMPONENTS]
    win_stds   = [np.std(norm_wins[c])    if norm_wins[c]   else 0 for c in COMPONENTS]
    loss_stds  = [np.std(norm_losses[c])  if norm_losses[c] else 0 for c in COMPONENTS]

    ax_main.bar(x - width/2, win_means,  width, label=f'Win (n={len(wins)})',
                color='#2ecc71', alpha=0.85, yerr=win_stds,  capsize=4, error_kw={'linewidth': 1})
    ax_main.bar(x + width/2, loss_means, width, label=f'Loss (n={len(losses)})',
                color='#e74c3c', alpha=0.85, yerr=loss_stds, capsize=4, error_kw={'linewidth': 1})

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(COMPONENT_LABELS, rotation=20, ha='right', fontsize=9)
    ax_main.set_ylabel("Normalized Value (0-1)")
    ax_main.set_title(f"{agent_type}\nAvg Reward Components: Wins vs Losses", fontsize=11, fontweight='bold')
    ax_main.legend(fontsize=9)
    ax_main.set_ylim(0, 1.4)
    ax_main.grid(axis='y', alpha=0.3)

    for i, (w, l) in enumerate(zip(win_means, loss_means)):
        delta = w - l
        sig   = ttest_sig(norm_wins[COMPONENTS[i]], norm_losses[COMPONENTS[i]])
        color = '#27ae60' if delta > 0 else '#c0392b'
        top   = max(w, l) + max(win_stds[i], loss_stds[i]) + 0.04
        ax_main.text(i, top,      f'{delta:+.2f}', ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold')
        ax_main.text(i, top+0.08, sig,             ha='center', va='bottom', fontsize=7,   color='#555')

    # ── Distribution per component ──────────────────────────────────
    ax_dist.set_visible(False)
    fig = ax_dist.get_figure()
    gs2 = ax_dist.get_subplotspec().subgridspec(1, len(COMPONENTS))
    dist_axes = [fig.add_subplot(gs2[0, i]) for i in range(len(COMPONENTS))]

    for i, (c, label) in enumerate(zip(COMPONENTS, COMPONENT_LABELS)):
        axi = dist_axes[i]
        data_to_plot, colors_bp, pos = [], [], []
        if wins_avg[c]:
            data_to_plot.append(wins_avg[c]); colors_bp.append('#2ecc71'); pos.append(1)
        if losses_avg[c]:
            data_to_plot.append(losses_avg[c]); colors_bp.append('#e74c3c'); pos.append(1.6)
        if data_to_plot:
            bp = axi.boxplot(data_to_plot, positions=pos, patch_artist=True,
                             widths=0.4, showfliers=False,
                             medianprops={'color': 'black', 'linewidth': 1.5})
            for patch, color in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(color); patch.set_alpha(0.75)
        axi.set_title(label, fontsize=8, fontweight='bold')
        axi.set_xticks([])
        axi.grid(axis='y', alpha=0.3)
        if i == 0:
            axi.set_ylabel("Raw Value", fontsize=8)
        else:
            axi.tick_params(labelleft=False)

    dist_axes[0].get_figure().text(
        0.5, ax_dist.get_position().y1 + 0.01,
        "Distribution per Component (green=win, red=loss)",
        ha='center', fontsize=10
    )

    # ── Final board state ──────────────────────────────────────────
    ax_final.set_visible(False)
    gs3 = ax_final.get_subplotspec().subgridspec(1, len(COMPONENTS))
    final_axes = [fig.add_subplot(gs3[0, i]) for i in range(len(COMPONENTS))]

    for i, (c, label) in enumerate(zip(COMPONENTS, COMPONENT_LABELS)):
        axi = final_axes[i]
        wm = np.mean(wins_final[c])   if wins_final[c]   else 0
        lm = np.mean(losses_final[c]) if losses_final[c] else 0
        axi.bar([0],   [wm], color='#2ecc71', alpha=0.85, width=0.4)
        axi.bar([0.5], [lm], color='#e74c3c', alpha=0.85, width=0.4)
        axi.set_title(label, fontsize=8, fontweight='bold')
        axi.set_xticks([0, 0.5])
        axi.set_xticklabels(['W', 'L'], fontsize=8)
        axi.grid(axis='y', alpha=0.3)
        if i == 0:
            axi.set_ylabel("Raw Value", fontsize=8)
        else:
            axi.tick_params(labelleft=False)

    final_axes[0].get_figure().text(
        0.5, ax_final.get_position().y1 + 0.01,
        "Final Board State Breakdown",
        ha='center', fontsize=10
    )

    print(f"\n  Key differences (win - loss, normalized avg):")
    deltas = [(COMPONENT_LABELS[i], win_means[i] - loss_means[i]) for i in range(len(COMPONENTS))]
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    for label, d in deltas:
        direction = "HIGHER in wins" if d > 0 else "HIGHER in losses"
        print(f"    {label:<18} {d:+.3f}  ({direction})")


def plot_correlation_and_scatter(runs: List[Dict], agent_type: str, results_dir: str):
    if not runs:
        return

    scores    = np.array([r['score'] for r in runs])
    won       = np.array([1 if r.get('reached_2048') or r.get('won') else 0 for r in runs])
    comp_vals = {c: [] for c in COMPONENTS}
    for r in runs:
        bd = r.get('avg_reward_breakdown', {})
        for c in COMPONENTS:
            val = bd.get(c, 0)
            if FLIP_SIGN.get(c):
                val = -val
            comp_vals[c].append(val)
    comp_vals = {c: np.array(v) for c, v in comp_vals.items()}

    # Correlation heatmap
    targets     = {'Final Score': scores, 'Win (0/1)': won}
    corr_matrix = np.zeros((len(COMPONENTS), len(targets)))
    for i, c in enumerate(COMPONENTS):
        for j, (_, tvals) in enumerate(targets.items()):
            if np.std(comp_vals[c]) > 0:
                corr_matrix[i, j] = np.corrcoef(comp_vals[c], tvals)[0, 1]

    fig1, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(list(targets.keys()), fontsize=10)
    ax.set_yticks(range(len(COMPONENTS)))
    ax.set_yticklabels(COMPONENT_LABELS, fontsize=10)
    ax.set_title(f"{agent_type}\nPearson Correlation with Outcome", fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    for i in range(len(COMPONENTS)):
        for j in range(len(targets)):
            ax.text(j, i, f'{corr_matrix[i, j]:+.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    plt.tight_layout()
    out1 = os.path.join(results_dir, "correlation.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Correlation plot saved -> {out1}")

    # Scatter plots
    fig2, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig2.suptitle(f"{agent_type} -- Component vs Final Score", fontsize=13, fontweight='bold')
    axes = axes.flatten()
    colors_scatter = ['#2ecc71' if w else '#e74c3c' for w in won]

    for i, (c, label) in enumerate(zip(COMPONENTS, COMPONENT_LABELS)):
        ax = axes[i]
        ax.scatter(comp_vals[c], scores, c=colors_scatter, alpha=0.65, s=40, edgecolors='none')
        if np.std(comp_vals[c]) > 0:
            z    = np.polyfit(comp_vals[c], scores, 1)
            xs   = np.linspace(comp_vals[c].min(), comp_vals[c].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), 'k--', linewidth=1.2, alpha=0.6)
            corr = np.corrcoef(comp_vals[c], scores)[0, 1]
            ax.set_title(f"{label}  (r={corr:+.2f})", fontsize=10, fontweight='bold')
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel("Avg Component Value", fontsize=8)
        ax.set_ylabel("Final Score", fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color='#2ecc71', label='Win'),
                                Patch(color='#e74c3c', label='Loss')], fontsize=8)

    plt.tight_layout()
    out2 = os.path.join(results_dir, "scatter.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Scatter plot saved -> {out2}")


def compare_agents(agent_types: List[str]):
    """Full cross-agent comparison: win rate, component deltas, win-only radar, score distribution."""

    all_runs   = {}
    all_wins   = {}
    all_losses = {}
    for agent in agent_types:
        runs = load_runs(agent)
        if not runs:
            continue
        wins, losses = split_wins_losses(runs)
        all_runs[agent]   = runs
        all_wins[agent]   = wins
        all_losses[agent] = losses

    agents = list(all_runs.keys())
    colors = AGENT_COLORS[:len(agents)]

    # ── Figure 1: Win rate + Component delta heatmap ─────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Cross-Agent Strategy Comparison", fontsize=13, fontweight='bold')

    win_rates             = [len(all_wins[a]) / len(all_runs[a]) * 100 for a in agents]
    component_win_deltas  = {c: [] for c in COMPONENTS}

    for agent in agents:
        wins_avg   = extract_component_values(all_wins[agent],   "avg_reward_breakdown")
        losses_avg = extract_component_values(all_losses[agent], "avg_reward_breakdown")
        norm_wins, norm_losses = normalize_across_all(wins_avg, losses_avg)
        for c in COMPONENTS:
            wm = np.mean(norm_wins[c])   if norm_wins[c]   else 0
            lm = np.mean(norm_losses[c]) if norm_losses[c] else 0
            component_win_deltas[c].append(wm - lm)

    agent_labels = [a.replace('_', '\n') for a in agents]
    axes[0].bar(agent_labels, win_rates, color=colors, alpha=0.85)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title("Win Rate by Agent")
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(win_rates):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    matrix = np.array([component_win_deltas[c] for c in COMPONENTS])
    im = axes[1].imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[1].set_xticks(range(len(agents)))
    axes[1].set_xticklabels(agent_labels, fontsize=9)
    axes[1].set_yticks(range(len(COMPONENTS)))
    axes[1].set_yticklabels(COMPONENT_LABELS, fontsize=9)
    axes[1].set_title("Win-Loss Component Delta\n(green = component higher in wins)")
    plt.colorbar(im, ax=axes[1], label='Normalized delta')
    for i in range(len(COMPONENTS)):
        for j in range(len(agents)):
            axes[1].text(j, i, f'{matrix[i, j]:+.2f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    out1 = os.path.join("results", "comparison_strategy.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nComparison saved -> {out1}")

    # ── Figure 2: Score distribution (all games + wins only) ─────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Score Distribution by Agent", fontsize=13, fontweight='bold')

    for ax, subset_label, subset_fn in [
        (axes2[0], "All Games",   lambda a: all_runs[a]),
        (axes2[1], "Wins Only",   lambda a: all_wins[a]),
    ]:
        data   = [np.array([r['score'] for r in subset_fn(a)]) for a in agents]
        parts  = ax.violinplot(data, positions=range(len(agents)), showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('black')
                parts[key].set_linewidth(1.2)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agent_labels, fontsize=9)
        ax.set_ylabel("Final Score")
        ax.set_title(subset_label)
        ax.grid(axis='y', alpha=0.3)
        for i, (a, d) in enumerate(zip(agents, data)):
            if len(d):
                ax.text(i, d.max() * 1.02, f'med={int(np.median(d))}',
                        ha='center', fontsize=7.5, color='#333')

    plt.tight_layout()
    out2 = os.path.join("results", "comparison_score_dist.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Score distribution saved -> {out2}")

    # ── Figure 3: Win-only component fingerprint (bar chart) ─────────
    # Normalize each component globally across all winning games of all agents
    all_win_vals = {c: [] for c in COMPONENTS}
    for agent in agents:
        d = extract_component_values(all_wins[agent], "avg_reward_breakdown")
        for c in COMPONENTS:
            all_win_vals[c].extend(d[c])

    global_ranges = {}
    for c in COMPONENTS:
        vals = all_win_vals[c]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        global_ranges[c] = (lo, hi - lo if hi != lo else 1.0)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    x     = np.arange(len(COMPONENTS))
    width = 0.8 / len(agents)

    for idx, (agent, color) in enumerate(zip(agents, colors)):
        d = extract_component_values(all_wins[agent], "avg_reward_breakdown")
        means = []
        stds  = []
        for c in COMPONENTS:
            lo, rng = global_ranges[c]
            normed  = [(v - lo) / rng for v in d[c]] if d[c] else [0]
            means.append(np.mean(normed))
            stds.append(np.std(normed))
        offset = (idx - (len(agents) - 1) / 2) * width
        bars   = ax3.bar(x + offset, means, width, label=agent.replace('_', ' '),
                         color=color, alpha=0.85, yerr=stds, capsize=3, error_kw={'linewidth': 1})

    ax3.set_xticks(x)
    ax3.set_xticklabels(COMPONENT_LABELS, rotation=15, ha='right', fontsize=10)
    ax3.set_ylabel("Normalized Value (0-1, globally scaled)")
    ax3.set_title("Winning Strategy Fingerprint: Component Profile in Winning Games", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 1.3)
    ax3.grid(axis='y', alpha=0.3)

    # Annotate difference between agents on each component
    if len(agents) == 2:
        d0 = extract_component_values(all_wins[agents[0]], "avg_reward_breakdown")
        d1 = extract_component_values(all_wins[agents[1]], "avg_reward_breakdown")
        for i, c in enumerate(COMPONENTS):
            lo, rng = global_ranges[c]
            n0 = [(v - lo) / rng for v in d0[c]] if d0[c] else [0]
            n1 = [(v - lo) / rng for v in d1[c]] if d1[c] else [0]
            diff = np.mean(n0) - np.mean(n1)
            sig  = ttest_sig(n0, n1)
            ax3.text(i, max(np.mean(n0), np.mean(n1)) + 0.15,
                     f'diff={diff:+.2f}\n{sig}', ha='center', fontsize=7.5, color='#333')

    plt.tight_layout()
    out3 = os.path.join("results", "comparison_win_fingerprint.png")
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Win fingerprint saved -> {out3}")

    # ── Figure 4: Side-by-side scatter — Merge Potential vs Tile Score ─
    # Most informative pair from earlier analysis
    fig4, axes4 = plt.subplots(1, len(agents), figsize=(7 * len(agents), 5), sharey=True)
    if len(agents) == 1:
        axes4 = [axes4]
    fig4.suptitle("Merge Potential vs Tile Score (green=win, red=loss)", fontsize=12, fontweight='bold')

    for ax, agent in zip(axes4, agents):
        runs = all_runs[agent]
        won  = np.array([1 if r.get('reached_2048') or r.get('won') else 0 for r in runs])
        mp   = np.array([r.get('avg_reward_breakdown', {}).get('merge_potential', 0) for r in runs])
        ts   = np.array([r.get('avg_reward_breakdown', {}).get('tile_score',      0) for r in runs])
        clrs = ['#2ecc71' if w else '#e74c3c' for w in won]
        ax.scatter(mp, ts, c=clrs, alpha=0.65, s=45, edgecolors='none')
        ax.set_xlabel("Avg Merge Potential", fontsize=10)
        ax.set_ylabel("Avg Tile Score",      fontsize=10)
        ax.set_title(agent.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        # Cluster centroids
        for mask, label, col in [(won == 1, 'win', '#27ae60'), (won == 0, 'loss', '#c0392b')]:
            if mask.sum():
                ax.scatter(mp[mask].mean(), ts[mask].mean(), marker='*', s=250,
                           color=col, edgecolors='black', linewidths=0.5, zorder=5, label=f'{label} centroid')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out4 = os.path.join("results", "comparison_merge_vs_tile.png")
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Merge vs Tile scatter saved -> {out4}")

    # ── Figure 5: Performance summary ────────────────────────────────
    # Win rate with 95% CI, avg scores (all/wins/losses), percentiles, CDF
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle("Performance Summary", fontsize=13, fontweight='bold')

    # Wilson 95% CI for win rate
    def wilson_ci(wins, n, z=1.96):
        if n == 0: return 0, 0
        p = wins / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return max(0, centre - margin), min(1, centre + margin)

    # -- Win rate with CI --
    ax = axes5[0, 0]
    wr_vals, wr_lo, wr_hi = [], [], []
    for a in agents:
        n = len(all_runs[a]); w = len(all_wins[a])
        wr_vals.append(w / n * 100)
        lo, hi = wilson_ci(w, n)
        wr_lo.append((w/n - lo) * 100)
        wr_hi.append((hi - w/n) * 100)
    bars = ax.bar(agent_labels, wr_vals, color=colors, alpha=0.85)
    ax.errorbar(range(len(agents)), wr_vals,
                yerr=[wr_lo, wr_hi], fmt='none', color='black', capsize=6, linewidth=1.5)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate (95% Wilson CI)")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(wr_vals):
        ax.text(i, v + wr_hi[i] + 1.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # -- Avg score: all / wins / losses grouped --
    ax = axes5[0, 1]
    x = np.arange(len(agents))
    w = 0.25
    avg_all  = [np.mean([r['score'] for r in all_runs[a]])    for a in agents]
    avg_wins = [np.mean([r['score'] for r in all_wins[a]])    if all_wins[a]   else 0 for a in agents]
    avg_loss = [np.mean([r['score'] for r in all_losses[a]])  if all_losses[a] else 0 for a in agents]
    ax.bar(x - w, avg_all,  w, label='All',    color='#95a5a6', alpha=0.85)
    ax.bar(x,     avg_wins, w, label='Wins',   color='#2ecc71', alpha=0.85)
    ax.bar(x + w, avg_loss, w, label='Losses', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(agent_labels, fontsize=9)
    ax.set_ylabel("Avg Final Score")
    ax.set_title("Average Score: All / Wins / Losses")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for i, (a, b, c_) in enumerate(zip(avg_all, avg_wins, avg_loss)):
        ax.text(i-w, a*1.01, f'{int(a):,}', ha='center', fontsize=7)
        ax.text(i,   b*1.01, f'{int(b):,}', ha='center', fontsize=7)
        ax.text(i+w, c_*1.01, f'{int(c_):,}', ha='center', fontsize=7)

    # -- Score percentiles --
    ax = axes5[1, 0]
    percentiles = [25, 50, 75, 90]
    x = np.arange(len(percentiles))
    w = 0.8 / len(agents)
    for idx, (a, color) in enumerate(zip(agents, colors)):
        scores_all = [r['score'] for r in all_runs[a]]
        pct_vals   = [np.percentile(scores_all, p) for p in percentiles]
        offset     = (idx - (len(agents)-1)/2) * w
        ax.bar(x + offset, pct_vals, w, label=a.replace('_',' '), color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in percentiles], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Score Percentiles (all games)")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # -- CDF of final scores --
    ax = axes5[1, 1]
    for a, color in zip(agents, colors):
        scores_sorted = np.sort([r['score'] for r in all_runs[a]])
        cdf = np.arange(1, len(scores_sorted)+1) / len(scores_sorted)
        ax.plot(scores_sorted, cdf, color=color, linewidth=2, label=a.replace('_',' '))
    ax.axvline(x=20480, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='2048 threshold (~20k)')
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Final Scores")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out5 = os.path.join("results", "comparison_performance.png")
    plt.savefig(out5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"Performance summary saved -> {out5}")


def run_analysis(agent_types: List[str]):
    for agent_type in agent_types:
        runs = load_runs(agent_type)
        if not runs:
            continue

        results_dir = os.path.join("results", agent_type)
        os.makedirs(results_dir, exist_ok=True)

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f"Strategy Analysis: {agent_type}", fontsize=14, fontweight='bold', y=1.01)
        gs = gridspec.GridSpec(3, 1, hspace=0.55)

        ax_main  = fig.add_subplot(gs[0])
        ax_dist  = fig.add_subplot(gs[1])
        ax_final = fig.add_subplot(gs[2])

        analyze_agent(agent_type, ax_main, ax_dist, ax_final)

        out = os.path.join(results_dir, "strategy.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Plot saved -> {out}")

        plot_correlation_and_scatter(runs, agent_type, results_dir)

    if len(agent_types) > 1:
        compare_agents(agent_types)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_strategy.py <agent_type>")
        print("  e.g. python analyze_strategy.py beam_search")
        print("  e.g. python analyze_strategy.py beam_search mcts")
        sys.exit(1)

    agents = sys.argv[1:]
    print(f"Analyzing agents: {agents}")
    run_analysis(agents)
