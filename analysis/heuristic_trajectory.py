"""
heuristic_trajectory.py
=======================

Visualise which reward components dominate at every point in a game.

Requires games logged with log_move_detail=True:

    logger = RunLogger(log_move_detail=True)
    module = InteractionModule(config, agent, logger=logger, num_workers=1)
    module.run(num_games=10)

Usage:
    python analysis/heuristic_trajectory.py --agent beam_search
    python analysis/heuristic_trajectory.py --agent mcts --run_id abc123
    python analysis/heuristic_trajectory.py --agent beam_search --game 0
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

LOG_DIR    = "logs"
OUTPUT_DIR = "results"

COMPONENTS = ["tile_score", "empty_bonus", "monotonicity",
              "corner_bonus", "merge_potential", "smoothness"]
LABELS     = ["Tile Score", "Empty Bonus", "Monotonicity",
              "Corner Bonus", "Merge Potential", "Smoothness"]
FLIP_SIGN  = {"smoothness": True}   # negate so higher = better for all

COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c"]
SMOOTHING_WINDOW = 20   # rolling average window (moves)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_games(agent_type: str, run_id_prefix: str = None):
    """Load games that have move_reward_breakdowns."""
    path = os.path.join(LOG_DIR, f"{agent_type}_runs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No log file found: {path}")

    all_runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            all_runs[g.get("run_id", "unknown")].append(g)

    if run_id_prefix:
        matching = {r: gs for r, gs in all_runs.items() if r.startswith(run_id_prefix)}
        if not matching:
            raise ValueError(f"No run found matching prefix: {run_id_prefix}")
        rid = next(iter(matching))
    else:
        rid = max(all_runs, key=lambda r: all_runs[r][0].get("timestamp", ""))

    games = all_runs[rid]
    games_with_moves = [g for g in games if g.get("move_reward_breakdowns")]
    if not games_with_moves:
        raise ValueError(
            f"Run {rid[:12]} has no move_reward_breakdowns. "
            "Re-run with RunLogger(log_move_detail=True) and num_workers=1."
        )
    print(f"Loaded run {rid[:12]}: {len(games_with_moves)}/{len(games)} games have move detail")
    return games_with_moves, rid


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def extract_trajectories(game: dict):
    """Return dict of component -> np.array of values across moves."""
    moves = game["move_reward_breakdowns"]
    traj = {}
    for comp in COMPONENTS:
        vals = np.array([m.get(comp, 0.0) for m in moves], dtype=float)
        if FLIP_SIGN.get(comp):
            vals = -vals
        traj[comp] = vals
    return traj


def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def interpolate_to_pct(arr: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """Resample a trajectory of arbitrary length to n_bins percentile points."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, arr)


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

def plot_single_game(traj: dict, game_idx: int, out_path: str):
    """Plot all components for one game, normalised."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n_moves = len(next(iter(traj.values())))
    x = np.arange(n_moves)

    for comp, label, color in zip(COMPONENTS, LABELS, COLORS):
        vals = normalize(traj[comp])
        smooth = rolling_mean(vals, SMOOTHING_WINDOW)
        ax.plot(x, smooth, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Move Number")
    ax.set_ylabel("Normalised Value (higher = better)")
    ax.set_title(f"Heuristic Trajectory — Game {game_idx} ({n_moves} moves)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_average_trajectory(games: list, out_path: str, title_suffix: str = ""):
    """Resample every game to 100 percentile points, then average."""
    if not games:
        print(f"  Skipped (no games): {out_path}")
        return
    N_BINS = 100
    stacked = {c: [] for c in COMPONENTS}

    for g in games:
        traj = extract_trajectories(g)
        for comp in COMPONENTS:
            normed = normalize(traj[comp])
            resampled = interpolate_to_pct(normed, N_BINS)
            stacked[comp].append(resampled)

    x = np.linspace(0, 100, N_BINS)
    fig, ax = plt.subplots(figsize=(14, 5))

    for comp, label, color in zip(COMPONENTS, LABELS, COLORS):
        arr = np.array(stacked[comp])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)

    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_xlabel("Game Progress (%)")
    ax.set_ylabel("Normalised Value (higher = better)")
    ax.set_title(f"Average Heuristic Trajectory ({len(games)} games){suffix}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_win_loss_diff(wins: list, losses: list, out_path: str, title_suffix: str = ""):
    """For each component, plot mean(wins) - mean(losses) over game progress.

    Positive region = component is higher in winning games at that phase.
    Negative region = component is lower in winning games (losses have more).
    """
    if not wins or not losses:
        print(f"  Skipped win/loss diff (need both wins and losses): {out_path}")
        return

    N_BINS = 100
    SMOOTH_W = 12

    def stack_games(games_list):
        s = {c: [] for c in COMPONENTS}
        for g in games_list:
            traj = extract_trajectories(g)
            for comp in COMPONENTS:
                # Normalize per game so scale doesn't dominate
                normed = normalize(traj[comp])
                s[comp].append(interpolate_to_pct(normed, N_BINS))
        return {c: np.array(v) for c, v in s.items()}

    win_stack  = stack_games(wins)
    loss_stack = stack_games(losses)

    x = np.linspace(0, 100, N_BINS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    fig.suptitle(
        f"Heuristic Advantage: Wins vs Losses{' — ' + title_suffix if title_suffix else ''}\n"
        "(positive = component is higher in winning games at that game phase)",
        fontsize=12, fontweight="bold",
    )

    for ax, comp, label, color in zip(axes.flat, COMPONENTS, LABELS, COLORS):
        win_mean  = win_stack[comp].mean(axis=0)
        loss_mean = loss_stack[comp].mean(axis=0)
        diff      = win_mean - loss_mean

        # Smooth for readability
        kernel = np.ones(SMOOTH_W) / SMOOTH_W
        diff_s = np.convolve(diff, kernel, mode="same")

        # Confidence: bootstrap-style std of diff
        # approximate: sqrt(var_win/n_win + var_loss/n_loss)
        var_diff = (win_stack[comp].var(axis=0) / len(wins) +
                    loss_stack[comp].var(axis=0) / len(losses))
        se = np.sqrt(var_diff)
        se_s = np.convolve(se, kernel, mode="same")

        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        ax.fill_between(x, diff_s - se_s, diff_s + se_s, alpha=0.2, color=color)
        ax.plot(x, diff_s, color=color, linewidth=2)

        # Shade regions
        ax.fill_between(x, 0, diff_s, where=(diff_s > 0), alpha=0.15, color="green",
                        label="Win advantage")
        ax.fill_between(x, 0, diff_s, where=(diff_s < 0), alpha=0.15, color="red",
                        label="Loss advantage")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Game Progress (%)", fontsize=8)
        ax.set_ylabel("Win − Loss (normalised)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

        # Add annotation for overall direction
        net = diff_s.mean()
        direction = "WIN+" if net > 0.005 else ("LOSS+" if net < -0.005 else "neutral")
        ax.text(0.98, 0.98, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold",
                color="green" if net > 0.005 else ("red" if net < -0.005 else "gray"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_phase_bars(wins: list, losses: list, out_path: str, title_suffix: str = ""):
    """Mean heuristic per component in 4 game phases, win vs loss.

    Phases: Early (0-25%), Mid-early (25-50%), Mid-late (50-75%), Late (75-100%).
    """
    if not wins or not losses:
        print(f"  Skipped phase bars (need both wins and losses): {out_path}")
        return

    N_BINS = 100
    PHASES = [(0, 25), (25, 50), (50, 75), (75, 100)]
    PHASE_LABELS = ["Early\n(0-25%)", "Mid-Early\n(25-50%)",
                    "Mid-Late\n(50-75%)", "Late\n(75-100%)"]

    def phase_means(games_list):
        """Return array (n_games, n_components, n_phases)."""
        all_vals = []
        for g in games_list:
            traj = extract_trajectories(g)
            game_phases = []
            for comp in COMPONENTS:
                normed    = normalize(traj[comp])
                resampled = interpolate_to_pct(normed, N_BINS)
                comp_phases = []
                for lo, hi in PHASES:
                    comp_phases.append(resampled[lo:hi].mean())
                game_phases.append(comp_phases)
            all_vals.append(game_phases)
        return np.array(all_vals)  # (n_games, n_components, n_phases)

    win_arr  = phase_means(wins)   # (nW, 6, 4)
    loss_arr = phase_means(losses) # (nL, 6, 4)

    win_mean  = win_arr.mean(axis=0)   # (6, 4)
    loss_mean = loss_arr.mean(axis=0)
    win_se    = win_arr.std(axis=0) / np.sqrt(len(wins))
    loss_se   = loss_arr.std(axis=0) / np.sqrt(len(losses))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    fig.suptitle(
        f"Heuristic Value by Game Phase: Wins vs Losses"
        f"{' — ' + title_suffix if title_suffix else ''}\n"
        "(green = wins, red = losses; higher = better for all components)",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(PHASES))
    bar_w = 0.35

    for ax, i_comp, label in zip(axes.flat, range(len(COMPONENTS)), LABELS):
        w_vals = win_mean[i_comp]
        l_vals = loss_mean[i_comp]
        w_err  = win_se[i_comp]
        l_err  = loss_se[i_comp]

        ax.bar(x - bar_w/2, w_vals, bar_w, color="green",   alpha=0.65,
               yerr=w_err, capsize=3, label=f"Wins (n={len(wins)})")
        ax.bar(x + bar_w/2, l_vals, bar_w, color="crimson", alpha=0.65,
               yerr=l_err, capsize=3, label=f"Losses (n={len(losses)})")

        ax.set_xticks(x)
        ax.set_xticklabels(PHASE_LABELS, fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Normalised value", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        if i_comp == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dominance_heatmap(games: list, out_path: str, title_suffix: str = ""):
    """Which component ranks #1 most often at each game-progress decile."""
    if not games:
        print(f"  Skipped (no games): {out_path}")
        return
    N_BINS = 50
    dominance = np.zeros((len(COMPONENTS), N_BINS))

    for g in games:
        traj = extract_trajectories(g)
        resampled = np.array([
            interpolate_to_pct(normalize(traj[c]), N_BINS)
            for c in COMPONENTS
        ])  # shape: (n_components, N_BINS)
        ranks = np.argsort(resampled, axis=0)[::-1]  # descending
        for bin_i in range(N_BINS):
            dominance[ranks[0, bin_i], bin_i] += 1

    dominance /= len(games)

    suffix = f" — {title_suffix}" if title_suffix else ""
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(dominance, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   extent=[0, 100, len(COMPONENTS) - 0.5, -0.5])
    ax.set_yticks(range(len(COMPONENTS)))
    ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Game Progress (%)")
    ax.set_title(f"Dominance Heatmap — Fraction of games where component ranks #1{suffix}")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",  required=True,
                        help="Agent type slug, e.g. beam_search, mcts")
    parser.add_argument("--run_id", default=None,
                        help="Run ID prefix to target a specific run")
    parser.add_argument("--game",   type=int, default=0,
                        help="Which game index to plot for the single-game chart (default: 0)")
    args = parser.parse_args()

    games, rid = load_games(args.agent, args.run_id)

    wins   = [g for g in games if g.get("reached_2048") or g.get("won")]
    losses = [g for g in games if not (g.get("reached_2048") or g.get("won"))]
    print(f"  Wins: {len(wins)}  Losses: {len(losses)}")

    out_dir = os.path.join(OUTPUT_DIR, args.agent)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Single game
    game_idx = min(args.game, len(games) - 1)
    traj = extract_trajectories(games[game_idx])
    plot_single_game(traj, game_idx,
                     os.path.join(out_dir, "heuristic_trajectory_single.png"))

    # 2. Average trajectory — all / wins / losses
    plot_average_trajectory(games,  os.path.join(out_dir, "heuristic_trajectory_all.png"),  "All Games")
    plot_average_trajectory(wins,   os.path.join(out_dir, "heuristic_trajectory_wins.png"),  "Wins")
    plot_average_trajectory(losses, os.path.join(out_dir, "heuristic_trajectory_losses.png"), "Losses")

    # 3. Dominance heatmap — all / wins / losses
    plot_dominance_heatmap(games,  os.path.join(out_dir, "heuristic_dominance_all.png"),    "All Games")
    plot_dominance_heatmap(wins,   os.path.join(out_dir, "heuristic_dominance_wins.png"),   "Wins")
    plot_dominance_heatmap(losses, os.path.join(out_dir, "heuristic_dominance_losses.png"), "Losses")

    # 4. Win vs Loss difference trajectory (6-panel)
    plot_win_loss_diff(wins, losses,
                       os.path.join(out_dir, "heuristic_win_loss_diff.png"),
                       title_suffix=args.agent)

    # 5. Phase-binned bar chart (6-panel)
    plot_phase_bars(wins, losses,
                    os.path.join(out_dir, "heuristic_phase_bars.png"),
                    title_suffix=args.agent)

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
