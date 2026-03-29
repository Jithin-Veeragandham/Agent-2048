"""
cma_tuner.py
============

Uses CMA-ES to optimize RewardFunction weights for BeamSearchAgent.

Requires: pip install cma

Usage:
    python cma_tuner.py
    python cma_tuner.py --resume    # resume from last checkpoint

Logs:
    logs/cma_tuning_log.jsonl    — per-generation results
    logs/cma_checkpoint.json     — latest state for resume
    logs/cma_best_weights.json   — best weights found so far
"""

import cma
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count

from game import Game2048, Action
from agents.beam_search import BeamSearchAgent
from framework.evaluation import RewardFunction


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
GAMES_PER_EVAL = 10
BEAM_WIDTH = 10
SEARCH_DEPTH = 5

INITIAL_WEIGHTS = [0.98, 2.23, 2.47, 1.42, 1.68, 2.56]
INITIAL_SIGMA = 0.3  # smaller since we're refining, not exploring

POPULATION_SIZE = 12
MAX_GENERATIONS = 100

NUM_WORKERS = max(1, cpu_count() - 1)  # leave one core free

WEIGHT_NAMES = ['tile', 'empty', 'mono', 'corner', 'merge', 'smooth']

# Wide bounds — let CMA-ES explore freely
LOWER_BOUNDS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UPPER_BOUNDS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "cma_tuning_log.jsonl"
CHECKPOINT_FILE = LOG_DIR / "cma_checkpoint.json"
BEST_FILE = LOG_DIR / "cma_best_weights.json"


# ═══════════════════════════════════════════════════════════════════
#  FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════════

def evaluate_weights(weight_vector):
    """Run games with given weights. Returns (fitness, detailed_stats).

    CMA-ES minimizes, so fitness = -score.
    """
    weights = {name: max(val, 0.0) for name, val in zip(WEIGHT_NAMES, weight_vector)}
    reward_fn = RewardFunction(weights=weights)
    agent = BeamSearchAgent(
        beam_width=BEAM_WIDTH,
        search_depth=SEARCH_DEPTH,
        reward_fn=reward_fn,
    )

    games = []

    for _ in range(GAMES_PER_EVAL):
        game = Game2048({"grid_size": 4})
        move_count = 0

        while not game.is_game_over():
            available = game.get_available_moves()
            if not available:
                break
            action = agent.choose_action(
                game.get_state(), available,
                game_context={"game": game}
            )
            game.move(action)
            move_count += 1

        board = game.get_state()
        breakdown = reward_fn.compute_breakdown(board)

        games.append({
            "score": game.get_score(),
            "highest_tile": int(np.max(board)),
            "moves": move_count,
            "final_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        })

    scores = [g["score"] for g in games]
    tiles = [g["highest_tile"] for g in games]
    win_count = sum(1 for t in tiles if t >= 2048)
    bonus = win_count * 5000

    stats = {
        "weights": {n: round(float(v), 4) for n, v in zip(WEIGHT_NAMES, weight_vector)},
        "avg_score": round(float(np.mean(scores)), 1),
        "max_score": int(max(scores)),
        "min_score": int(min(scores)),
        "std_score": round(float(np.std(scores)), 1),
        "avg_highest_tile": float(np.mean(tiles)),
        "max_highest_tile": int(max(tiles)),
        "tile_distribution": {str(t): tiles.count(t) for t in sorted(set(tiles))},
        "win_rate": round(win_count / GAMES_PER_EVAL * 100, 1),
        "avg_moves": round(float(np.mean([g["moves"] for g in games])), 1),
        # Average of final board breakdowns across games — shows what
        # the heuristic components look like at game end
        "avg_final_breakdown": _avg_breakdowns([g["final_breakdown"] for g in games]),
        "per_game": games,
    }

    return -(float(np.mean(scores)) + bonus), stats


def _avg_breakdowns(breakdowns):
    """Average multiple breakdown dicts."""
    if not breakdowns:
        return {}
    keys = breakdowns[0].keys()
    return {k: round(float(np.mean([b[k] for b in breakdowns])), 4) for k in keys}


# ═══════════════════════════════════════════════════════════════════
#  CHECKPOINT / RESUME
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(gen, best_weights, best_fitness, best_stats, es_mean, es_sigma):
    """Save current state so we can resume after crash."""
    checkpoint = {
        "generation": gen,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_weights": {n: round(float(v), 4) for n, v in zip(WEIGHT_NAMES, best_weights)},
        "best_fitness": float(best_fitness),
        "best_avg_score": best_stats["avg_score"],
        "es_mean": [float(x) for x in es_mean],
        "es_sigma": float(es_sigma),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def save_best(best_weights, best_stats, gen):
    """Save best weights to a standalone file for easy loading."""
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation": gen,
        "weights": {n: round(float(v), 4) for n, v in zip(WEIGHT_NAMES, best_weights)},
        "avg_score": best_stats["avg_score"],
        "max_score": best_stats["max_score"],
        "win_rate": best_stats["win_rate"],
        "max_highest_tile": best_stats["max_highest_tile"],
        "tile_distribution": best_stats["tile_distribution"],
        "usage": f"RewardFunction(weights={best_stats['weights']})",
    }
    with open(BEST_FILE, "w") as f:
        json.dump(result, f, indent=2)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if not CHECKPOINT_FILE.exists():
        return None
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    # Determine starting point
    initial_weights = INITIAL_WEIGHTS
    initial_sigma = INITIAL_SIGMA
    start_gen = 0

    if args.resume:
        ckpt = load_checkpoint()
        if ckpt:
            initial_weights = ckpt["es_mean"]
            initial_sigma = ckpt["es_sigma"]
            start_gen = ckpt["generation"]
            print(f"Resuming from generation {start_gen}")
            print(f"  Mean: {[round(x, 4) for x in initial_weights]}")
            print(f"  Sigma: {initial_sigma:.4f}")
        else:
            print("No checkpoint found, starting fresh")

    print("=" * 60)
    print("  CMA-ES Weight Tuner for BeamSearch 2048")
    print(f"  Population: {POPULATION_SIZE} | Games/eval: {GAMES_PER_EVAL}")
    print(f"  Beam: w={BEAM_WIDTH}, d={SEARCH_DEPTH}")
    print(f"  Workers: {NUM_WORKERS} | Max generations: {MAX_GENERATIONS}")
    print(f"  Weights: {WEIGHT_NAMES}")
    print(f"  Starting: {[round(x, 4) for x in initial_weights]}")
    print("=" * 60)

    # Initialize CMA-ES
    opts = cma.CMAOptions()
    opts.set("popsize", POPULATION_SIZE)
    opts.set("maxiter", MAX_GENERATIONS - start_gen)
    opts.set("bounds", [LOWER_BOUNDS, UPPER_BOUNDS])
    opts.set("tolfun", 1e-3)
    opts.set("verbose", -1)

    es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma, opts)

    best_ever_fitness = float("inf")
    best_ever_weights = None
    best_ever_stats = None
    gen = start_gen

    while not es.stop():
        gen += 1
        gen_start = time.time()

        candidates = es.ask()

        # Evaluate candidates in parallel
        t_eval = time.time()
        with Pool(processes=min(NUM_WORKERS, len(candidates))) as pool:
            results = pool.map(evaluate_weights, candidates)
        eval_elapsed = time.time() - t_eval

        fitnesses = [r[0] for r in results]
        all_stats = [r[1] for r in results]

        for i, stats in enumerate(all_stats):
            print(f"  Gen {gen:3d} | {i+1:2d}/{len(candidates)} | "
                  f"avg={stats['avg_score']:8.0f} | "
                  f"max_tile={stats['max_highest_tile']:5d} | "
                  f"wins={stats['win_rate']:5.1f}% | {stats['weights']}")

        es.tell(candidates, fitnesses)

        gen_elapsed = time.time() - gen_start
        gen_best_idx = int(np.argmin(fitnesses))
        gen_worst_idx = int(np.argmax(fitnesses))

        if fitnesses[gen_best_idx] < best_ever_fitness:
            best_ever_fitness = fitnesses[gen_best_idx]
            best_ever_weights = candidates[gen_best_idx].copy()
            best_ever_stats = all_stats[gen_best_idx]
            save_best(best_ever_weights, best_ever_stats, gen)

        # Log full generation results
        gen_log = {
            "type": "generation",
            "generation": gen,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_sec": round(gen_elapsed, 1),
            # Best this generation
            "gen_best": {
                "weights": all_stats[gen_best_idx]["weights"],
                "avg_score": all_stats[gen_best_idx]["avg_score"],
                "win_rate": all_stats[gen_best_idx]["win_rate"],
                "tile_dist": all_stats[gen_best_idx]["tile_distribution"],
                "avg_final_breakdown": all_stats[gen_best_idx]["avg_final_breakdown"],
            },
            # Worst this generation (shows what NOT to do)
            "gen_worst": {
                "weights": all_stats[gen_worst_idx]["weights"],
                "avg_score": all_stats[gen_worst_idx]["avg_score"],
                "win_rate": all_stats[gen_worst_idx]["win_rate"],
                "tile_dist": all_stats[gen_worst_idx]["tile_distribution"],
                "avg_final_breakdown": all_stats[gen_worst_idx]["avg_final_breakdown"],
            },
            # Population stats — shows convergence
            "population": {
                "avg_score_mean": round(float(np.mean([-f for f in fitnesses])), 1),
                "avg_score_std": round(float(np.std([-f for f in fitnesses])), 1),
                "cma_sigma": round(float(es.sigma), 4),
            },
            # Best ever
            "best_ever": {
                "weights": best_ever_stats["weights"],
                "avg_score": best_ever_stats["avg_score"],
                "win_rate": best_ever_stats["win_rate"],
                "found_at_gen": gen if fitnesses[gen_best_idx] == best_ever_fitness else "earlier",
            },
        }

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(gen_log) + "\n")

        save_checkpoint(gen, best_ever_weights, best_ever_fitness,
                        best_ever_stats, es.mean, es.sigma)

        # Print generation summary
        print(f"\n{'─' * 60}")
        print(f"  Gen {gen} | {gen_elapsed:.0f}s | sigma={es.sigma:.4f}")
        print(f"  Best this gen:  {all_stats[gen_best_idx]['avg_score']:.0f} "
              f"(wins {all_stats[gen_best_idx]['win_rate']}%)")
        print(f"  Worst this gen: {all_stats[gen_worst_idx]['avg_score']:.0f} "
              f"(wins {all_stats[gen_worst_idx]['win_rate']}%)")
        print(f"  Population avg: {gen_log['population']['avg_score_mean']:.0f} "
              f"± {gen_log['population']['avg_score_std']:.0f}")
        print(f"  Best ever:      {best_ever_stats['avg_score']:.0f} "
              f"| {best_ever_stats['weights']}")
        print(f"{'─' * 60}\n")

    # ── Final report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TUNING COMPLETE")
    print("=" * 60)
    print(f"  Generations:     {gen}")
    print(f"  Best avg score:  {best_ever_stats['avg_score']:.0f}")
    print(f"  Best win rate:   {best_ever_stats['win_rate']}%")
    print(f"  Best max tile:   {best_ever_stats['max_highest_tile']}")
    print(f"  Tile dist:       {best_ever_stats['tile_distribution']}")
    print(f"\n  Best weights:")
    for k, v in best_ever_stats["weights"].items():
        print(f"    {k:>8}: {v}")
    print(f"\n  Usage:")
    print(f"    RewardFunction(weights={best_ever_stats['weights']})")
    print(f"\n  Logs:     {LOG_FILE}")
    print(f"  Best:     {BEST_FILE}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
