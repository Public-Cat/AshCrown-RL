#!/usr/bin/env python3
"""Long-episode **run-mode** smoke test (RL-5, ``docs/07`` §4.1/§7.3).

Drives the scripted heuristic (``ashcrown_rl.baseline``) through a real
``AshCrownEnv(mode="run")`` episode — the full ten-level descent — and reports
how far it got: the levels reached, the terminal reason, the kills/gold and the
summed reward components, plus the decision throughput.

::

    uv run scripts/run_mode_smoke.py --decisions 20000 --seed 20260101
    uv run scripts/run_mode_smoke.py --decisions 20000 --json /tmp/run.json

Unlike the GUT scripted clear harness (which accelerates combat and therefore
always reaches victory) this is an **honest** policy rollout: the heuristic only
acts through the action space, so the level it stops on is the policy's real
run-mode result. ``--decisions`` and ``--time-budget`` bound the wall clock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402  (path set up above)

from ashcrown_rl import AshCrownEnv, HeuristicPolicy  # noqa: E402
from ashcrown_rl.env import DEFAULT_RUN_MAX_TICKS, MERCHANT_POLICIES  # noqa: E402
from ashcrown_rl.launch import default_game_project, find_godot  # noqa: E402

#: Reward components summed over the episode.
COMPONENTS = (
    "damage_dealt",
    "damage_taken",
    "kill_normal",
    "kill_elite",
    "kill_boss",
    "level_clear",
    "portal_entered",
    "victory",
    "death",
    "step_penalty",
    "gold",
    "gem",
    "heal",
    "whiff",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument(
        "--decisions",
        type=int,
        default=20_000,
        help="decision windows to step at most (15 per game-second at frame_skip=4)",
    )
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_RUN_MAX_TICKS)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--character", default="warrior")
    parser.add_argument(
        "--merchant-policy",
        default="auto_heal",
        choices=sorted(MERCHANT_POLICIES),
        help="engine-side merchant sub-policy (the economy stays out of the action space)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=180.0,
        help="stop after this many wall-clock seconds (0 = no limit)",
    )
    parser.add_argument("--godot-bin", default=None)
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--json", default=None, help="write the summary JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="only print the JSON summary")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    project = args.game_project or default_game_project()
    godot = args.godot_bin or find_godot()
    if project is None or godot is None:  # pragma: no cover - environment guard
        raise SystemExit(
            "no game checkout or godot binary found; set ASHCROWN_PROJECT / ASHCROWN_GODOT"
        )
    env = AshCrownEnv(
        mode="run",
        level=1,
        character=args.character,
        frame_skip=args.frame_skip,
        max_ticks=args.max_ticks,
        merchant_policy=args.merchant_policy,
        godot_bin=godot,
        project=str(project),
    )
    totals = dict.fromkeys(COMPONENTS, 0.0)
    levels: list[int] = []
    events_total: dict[str, float] = {
        "kills": 0.0,
        "gold_gained": 0.0,
        "merchants_bought": 0.0,
        "level_clears": 0.0,
    }
    started = time.perf_counter()
    decisions = 0
    terminal_reason = ""
    terminated = truncated = False
    level_started = time.perf_counter()
    per_level: list[dict[str, Any]] = []
    policy_resets = 0
    try:
        policy = HeuristicPolicy(env.layout)
        obs, info = env.reset(seed=args.seed)
        levels.append(int(info["level"]))
        while decisions < args.decisions:
            action = policy(obs, info, env.layout)
            obs, reward, terminated, truncated, info = env.step(action)
            decisions += 1
            for key in COMPONENTS:
                totals[key] += float(info.get("reward_components", {}).get(key, 0.0))
            events = info.get("events") or {}
            events_total["kills"] += float(events.get("kills", 0))
            events_total["gold_gained"] += float(events.get("gold_gained", 0))
            events_total["merchants_bought"] += float(events.get("merchant_bought", 0))
            if int(events.get("level_cleared", 0)) > 0:
                events_total["level_clears"] += 1.0
            if int(info["level"]) != levels[-1]:
                per_level.append(
                    {
                        "level": levels[-1],
                        "decisions": decisions,
                        "game_seconds": round(decisions * args.frame_skip / 60.0, 1),
                        "seconds": round(time.perf_counter() - level_started, 2),
                        "kills": int(info.get("kills", 0)),
                    }
                )
                levels.append(int(info["level"]))
                level_started = time.perf_counter()
                # The heuristic's navigation map is portal-relative and its
                # memories describe the floor it was built on, so a descend needs
                # a fresh policy (`HeuristicPolicy` is per floor by design).
                policy = HeuristicPolicy(env.layout)
                policy_resets += 1
            terminated = bool(terminated)
            truncated = bool(truncated)
            if terminated or truncated:
                terminal_reason = str(info.get("terminal_reason", ""))
                break
            if args.time_budget > 0 and time.perf_counter() - started > args.time_budget:
                terminal_reason = terminal_reason or "time_budget"
                break
        elapsed = time.perf_counter() - started
        summary: dict[str, Any] = {
            "mode": env.mode,
            "seed": args.seed,
            "merchant_policy": args.merchant_policy,
            "frame_skip": args.frame_skip,
            "decisions": decisions,
            "game_seconds": round(decisions * args.frame_skip / 60.0, 1),
            "elapsed_s": round(elapsed, 2),
            "decisions_per_s": round(decisions / elapsed, 1) if elapsed > 0 else 0.0,
            "levels_reached": levels,
            "levels_cleared": len(levels) - 1,
            "policy_resets": policy_resets,
            "per_level": per_level,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": terminal_reason,
            "kills": int(info.get("kills", 0)),
            "gold": int(info.get("gold", 0)),
            "events": events_total,
            "reward_components": {key: round(value, 4) for key, value in totals.items()},
            "reward_total": round(sum(totals.values()), 4),
            "layout_hash": env.layout_hash,
            "final_state_hash": str(info.get("state_hash", "")),
            "obs_len": env.obs_len,
            "obs_finite": bool(np.isfinite(obs).all()),
        }
    finally:
        env.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    if not args.quiet:
        print(
            "run mode: "
            f"{summary['decisions']} decisions ({summary['game_seconds']}s game, "
            f"{summary['elapsed_s']}s wall, {summary['decisions_per_s']}/s)"
        )
        print(
            f"levels reached: {summary['levels_reached']} "
            f"(cleared {summary['levels_cleared']}), "
            f"kills {summary['kills']}, gold {summary['gold']}"
        )
        for entry in summary["per_level"]:
            print(
                f"  left level {entry['level']} after {entry['decisions']} decisions "
                f"({entry['game_seconds']}s game, {entry['seconds']}s wall, "
                f"{entry['kills']} kills)"
            )
        print(
            "terminal: "
            f"{summary['terminal_reason'] or 'none'} "
            f"(terminated={summary['terminated']}, truncated={summary['truncated']})"
        )
        components = ", ".join(
            f"{key}={value}" for key, value in summary["reward_components"].items() if value
        )
        print(f"reward {summary['reward_total']}: {components or 'none'}")
        print(f"events: {summary['events']}")
    print("RUN_SMOKE " + json.dumps(summary, sort_keys=True))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
