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

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
