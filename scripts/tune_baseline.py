"""Fixed-seed tuning sweep for the scripted baseline (RL-4, rounds 3-4).

Runs the same 10-seed acceptance protocol as
``python -m ashcrown_rl.evaluate`` for a list of :class:`HeuristicConfig`
variants and prints one row per variant, so tuning is measured instead of
guessed::

    uv run python -m scripts.tune_baseline baseline dodge_only recover

Each variant is a name plus comma-separated ``field=value`` overrides on
:class:`~ashcrown_rl.baseline.HeuristicConfig`. The default variant list lives in
``VARIANTS`` below and is the tuning log's source of truth.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Any

from ashcrown_rl.baseline import DEFAULT_CONFIG, HeuristicPolicy
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.evaluate import run_episode

#: name -> HeuristicConfig overrides. The values in
#: ``docs/artifacts/RL/RL4-report.md`` §"RL-4 round 3" are the measured history;
#: this table lists the variants that still compile against the shipped config.
VARIANTS: dict[str, dict[str, Any]] = {
    # -- shipped configuration ------------------------------------------------ #
    "baseline": {},
    # -- round 3 knobs (kept: they still compile against the shipped config) --- #
    "keepout16": {"keepout_slack_px": 16.0},
    "keepout20": {"keepout_slack_px": 20.0},
    "margin20": {"strike_margin_px": 20.0},
    "margin40": {"strike_margin_px": 40.0},
    "margin50": {"strike_margin_px": 50.0},
    "margin70": {"strike_margin_px": 70.0},
    "margin40_keepout16": {"strike_margin_px": 40.0, "keepout_slack_px": 16.0},
    "standoff_high": {"standoff_frac": 1.05},
    "standoff_low": {"standoff_frac": 0.75},
    "no_sidestep": {"pack_sidestep_decisions": 0},
    "sidestep_wide": {"pack_sidestep_px": 90.0},
    "no_doorway_retreat": {"doorway_retreat": False},
    "no_nav": {"local_approach_px": 600.0},
    "no_goal_memory": {"goal_memory_decisions": 0},
    # -- round 4 pack-fighting micro, A/B one piece at a time ----------------- #
    # The round-3 policy with every round-4 flag off (the acceptance baseline
    # the last report shipped: 4/10 clears).
    "round3": {
        "strafe_dodge": False,
        "chokepoint": False,
        "isolate_target": False,
        "pack_gate": False,
        "queue_gate": False,
        "reaction_s": 0.0,
    },
    # Perpendicular strafe only (the shipped dodge).
    "dodge_only": {
        "chokepoint": False,
        "isolate_target": False,
        "pack_gate": False,
        "queue_gate": False,
        "reaction_s": 0.0,
    },
    # Strafe + the pack damage gate (refuse a swing a second attacker can punish).
    "pack_gate": {"strafe_dodge": False, "chokepoint": False, "isolate_target": False},
    # Strafe + isolate the most isolated target.
    "iso_only": {"strafe_dodge": False, "chokepoint": False, "pack_gate": False},
    # Strafe + fight from the nearest doorway when 2+ attackers are on the body.
    "choke_only": {"strafe_dodge": False, "isolate_target": False, "pack_gate": False},
    "dodge_choke": {"isolate_target": False, "pack_gate": False, "queue_gate": False},
    # Strafe + queue gate (swing only when every other attacker is a body behind).
    "queue": {"isolate_target": False, "pack_gate": False},
    # Strafe + punish only a telegraph/recovery (never a chasing enemy).
    "recover": {
        "chokepoint": False,
        "isolate_target": False,
        "pack_gate": False,
        "queue_gate": False,
        "punish_recover": True,
    },
    "dodge_react0": {"reaction_s": 0.0},
    "dodge_side0": {"strafe_side": 0},
    "hold_off": {"strafe_hold_bonus": 0.0},
    "hold_free": {"strafe_hold_floor_px": 999.0},
    "dodge35": {"dodge_window_s": 0.35},
    "dodge60": {"dodge_window_s": 0.6},
    "band16": {"strafe_band_px": 16.0},
    "keepout18": {"keepout_slack_px": 18.0},
    "margin60_dodge60": {"strike_margin_px": 60.0, "dodge_window_s": 0.6},
    "dodge_hold55": {"standoff_min_px": 55.0, "chokepoint": False},
    "dodge_hold62": {"standoff_min_px": 62.0, "chokepoint": False},
}


def run_variant(
    name: str,
    overrides: dict[str, Any],
    *,
    seeds: list[int],
    level: int,
    frame_skip: int,
    max_ticks: int,
    game_project: str | None,
) -> dict[str, Any]:
    """Run one variant over every seed and return its aggregate row."""
    config = replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG
    env = AshCrownEnv(
        level=level,
        frame_skip=frame_skip,
        max_ticks=max_ticks,
        project=game_project,
    )
    rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            policy = HeuristicPolicy(env.layout, config)
            metrics = run_episode(
                env,
                seed=seed,
                policy=policy,
                policy_name=f"heuristic:{name}",
                level=level,
                record_hashes=False,
            )
            rows.append(
                {
                    "seed": seed,
                    "reason": metrics.terminal_reason,
                    "seconds": round(metrics.game_seconds, 1),
                    "kills": metrics.kills,
                    "hp": round(metrics.player_hp, 1),
                    "taken": round(metrics.damage_taken, 1),
                    "swings": metrics.attacks_started,
                    "hits": metrics.hits_landed,
                }
            )
    finally:
        env.close()
    clears = sum(1 for row in rows if row["reason"] == "level_clear")
    deaths = sum(1 for row in rows if row["reason"] == "death")
    return {
        "variant": name,
        "clears": clears,
        "deaths": deaths,
        "truncated": len(rows) - clears - deaths,
        "kills": round(sum(row["kills"] for row in rows) / max(len(rows), 1), 2),
        "seconds": round(sum(row["seconds"] for row in rows) / max(len(rows), 1), 1),
        "swings": sum(row["swings"] for row in rows),
        "hits": sum(row["hits"] for row in rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", help="variant names (default: all)")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--detail", action="store_true", help="print the per-seed rows")
    args = parser.parse_args(argv)
    names = args.variants or list(VARIANTS)
    seeds = [args.seed_start + offset for offset in range(args.seed_count)]
    header = ("variant", "clears", "deaths", "trunc", "kills", "seconds", "swings", "hits")
    print("{:<20}{:>7}{:>7}{:>6}{:>7}{:>9}{:>7}{:>6}".format(*header))
    for name in names:
        if name not in VARIANTS:
            print(f"unknown variant {name!r}", file=sys.stderr)
            return 2
        result = run_variant(
            name,
            VARIANTS[name],
            seeds=seeds,
            level=args.level,
            frame_skip=args.frame_skip,
            max_ticks=args.max_ticks,
            game_project=args.game_project,
        )
        print(
            "{:<20}{:>7}{:>7}{:>6}{:>7}{:>9}{:>7}{:>6}".format(
                result["variant"],
                result["clears"],
                result["deaths"],
                result["truncated"],
                result["kills"],
                result["seconds"],
                result["swings"],
                result["hits"],
            )
        )
        if args.detail:
            for row in result["rows"]:
                print("    " + " ".join(f"{key}={value}" for key, value in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
