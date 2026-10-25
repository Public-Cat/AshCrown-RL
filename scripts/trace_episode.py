"""Per-decision trace of one heuristic episode (RL-4 round 3 measurement tool).

Writes one CSV row per decision window so the combat economics can be measured
instead of guessed: where the time goes (perceived target or not), at what
distance the committed swings start, what the target was doing at that moment,
and what the window produced (``attacks_started`` / ``hits_landed`` / damage).

::

    uv run python scripts/trace_episode.py --seed 2 --out /tmp/trace.csv
    uv run python scripts/trace_episode.py --seed 2 --summary

The summary aggregates the trace: decisions by branch (perceived / no target),
swings by enemy state-at-swing, swings by distance bucket, and the true
``hits_landed / attacks_started`` land rate.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.evaluate import resolve_policy
from ashcrown_rl.spaces import Layout

COLUMNS = (
    "decision",
    "seconds",
    "hp",
    "player_state",
    "attack_progress",
    "enemies_alive",
    "kills",
    "portal_active",
    "target_present",
    "target_dist_px",
    "facing_cos",
    "target_state",
    "target_windup_s",
    "target_cooldown_s",
    "target_reach_px",
    "target_hp_frac",
    "move",
    "attack",
    "interact",
    "started",
    "hits",
    "dealt",
    "taken",
)


def _facing_cos(facing: tuple[float, float], screen_cells: Any) -> float:
    """cos(angle) between the player's facing and the direction to the target."""
    fx, fy = float(facing[0]), float(facing[1])
    norm = float(np.hypot(fx, fy))
    if norm <= 1e-9:
        return 0.0
    tx = float(screen_cells[0] - screen_cells[1])
    ty = (float(screen_cells[0]) + float(screen_cells[1])) * 0.5
    tnorm = float(np.hypot(tx, ty))
    if tnorm <= 1e-9:
        return 1.0
    return (fx / norm) * (tx / tnorm) + (fy / norm) * (ty / tnorm)


def _distance_bucket(px: float) -> str:
    if px < 20.0:
        return "00-20"
    if px < 40.0:
        return "20-40"
    if px < 55.0:
        return "40-55"
    if px < 70.0:
        return "55-70"
    if px < 100.0:
        return "70-100"
    if px < 150.0:
        return "100-150"
    return "150+"


def trace(seed: int, level: int, frame_skip: int, max_ticks: int, out: Path | None) -> None:
    env = AshCrownEnv(level=level, frame_skip=frame_skip, max_ticks=max_ticks)
    rows: list[dict[str, Any]] = []
    try:
        layout: Layout = env.layout
        obs, info = env.reset(seed=seed, options={"level": level})
        policy = resolve_policy("heuristic", seed, layout)
        while True:
            player = layout.player_features(obs)
            enemies = [slot for slot in layout.perceived_enemies(obs) if not slot.dead]
            target = min(enemies, key=lambda slot: slot.screen_distance_px) if enemies else None
            action = policy(obs, info, layout)
            row: dict[str, Any] = {
                "decision": len(rows),
                "seconds": round(float(info.get("game_seconds", 0.0)), 3),
                "hp": round(player.hp, 1),
                "player_state": player.state,
                "attack_progress": round(player.attack_progress, 3),
                "enemies_alive": layout.enemies_alive(obs),
                "kills": layout.kills(obs),
                "portal_active": int(layout.portal_active(obs)),
                "target_present": 0 if target is None else 1,
                "target_dist_px": "" if target is None else round(target.screen_distance_px, 1),
                "facing_cos": (
                    ""
                    if target is None
                    else round(_facing_cos(player.facing, target.screen_delta_cells), 3)
                ),
                "target_state": "" if target is None else target.state,
                "target_windup_s": ("" if target is None else round(target.windup_remaining_s, 3)),
                "target_cooldown_s": ("" if target is None else round(target.attack_cooldown_s, 3)),
                "target_reach_px": "" if target is None else round(target.attack_reach_px, 1),
                "target_hp_frac": "" if target is None else round(target.hp_frac, 3),
                "move": int(action[0]),
                "attack": int(action[1]),
                "interact": int(action[2]),
            }
            obs, _reward, terminated, truncated, info = env.step(action)
            events = info.get("events") or {}
            row["started"] = int(events.get("attacks_started", 0))
            row["hits"] = int(events.get("hits_landed", 0))
            row["dealt"] = round(float(events.get("damage_dealt", 0.0)), 1)
            row["taken"] = round(float(events.get("damage_taken", 0.0)), 1)
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    _summarise(rows)


def _summarise(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    perceived = sum(1 for row in rows if row["target_present"])
    swings = _attribute_hits(rows)
    hits = sum(int(row["hits"]) for row in rows)
    started = sum(int(row["started"]) for row in rows)
    requests = sum(1 for row in rows if row["attack"] == 1)
    print(f"decisions={total} perceived={perceived} ({perceived / max(total, 1):.1%})")
    print(f"swings={started} hits={hits} requests={requests}")
    print(
        f"hits_per_swing={hits / started if started else 0.0:.3f} "
        f"hits_per_request={hits / requests if requests else 0.0:.3f} "
        f"hits_per_decision={hits / max(total, 1):.3f}"
    )
    by_state = Counter(row["target_state"] for row in swings)
    print("swing target state: " + ", ".join(f"{k}={v}" for k, v in sorted(by_state.items())))
    _rate_table("distance", swings, _distance_bucket, lambda row: row["target_dist_px"])
    _rate_table("facing cos", swings, _cos_bucket, lambda row: row["facing_cos"])
    by_player = Counter(row["player_state"] for row in rows)
    print("player state decisions: " + ", ".join(f"{k}={v}" for k, v in sorted(by_player.items())))
    in_reach = sum(
        1 for row in rows if row["target_present"] and float(row["target_dist_px"]) <= 65.0
    )
    taken = sum(float(row["taken"] or 0.0) for row in rows)
    dealt = sum(float(row["dealt"] or 0.0) for row in rows)
    print(f"decisions with a target within 65px: {in_reach}")
    print(f"damage dealt={dealt:.0f} taken={taken:.0f}")


def _attribute_hits(rows: list[dict[str, Any]], window: int = 10) -> list[dict[str, Any]]:
    """Rows that started a swing, with the hits that swing landed attached.

    A swing's hitbox goes live ~0.26 s (≈4 decisions) after the swing starts and
    stays live ~0.23 s (≈3 decisions), so a hit is attributed to the most recent
    swing start within ``window`` decisions. ``started``/``hits`` on the row are
    then swing-centric ("this swing started / this swing landed N hits").
    """
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if int(row["started"]) > 0:
            record = dict(row)
            record["swings"] = int(row["started"])
            record["hits"] = 0
            out.append(record)
        for record in out[::-1]:
            age = index - int(record["decision"])
            if age > window:
                break
            record["hits"] = int(record["hits"]) + int(row["hits"])
    return out


def _cos_bucket(value: float) -> str:
    cos = float(value)
    if cos < 0.5:
        return "<0.5"
    if cos < 0.75:
        return "0.5-0.75"
    if cos < 0.9:
        return "0.75-0.9"
    return ">=0.9"


def _rate_table(
    label: str,
    rows: list[dict[str, Any]],
    bucket_of: Any,
    value_of: Any,
) -> None:
    """Swing/hit counts and land rate per bucket of the value at swing start."""
    buckets: dict[str, list[int]] = {}
    for row in rows:
        if int(row["swings"]) <= 0 or value_of(row) == "":
            continue
        entry = buckets.setdefault(bucket_of(value_of(row)), [0, 0])
        entry[0] += int(row["swings"])
        entry[1] += int(row["hits"])
    parts = [
        f"{key}: {hits}/{swings} ({hits / swings:.0%})"
        for key, (swings, hits) in sorted(buckets.items())
    ]
    print(f"land rate by {label}: " + " | ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    trace(
        seed=args.seed,
        level=args.level,
        frame_skip=args.frame_skip,
        max_ticks=args.max_ticks,
        out=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
