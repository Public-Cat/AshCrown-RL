"""Per-decision trace and survival measurement of one heuristic episode (RL-4).

Writes one CSV row per decision window so the combat economics can be measured
instead of guessed: where the time goes (perceived target or not), at what
distance the committed swings start, what the target was doing at that moment,
and what the window produced (``attacks_started`` / ``hits_landed`` / damage).

Beyond the swing accounting it also runs the two **survival** measurements the
round-4 tuning needed (``docs/artifacts/RL/RL4-report.md``):

* **speed** — per-decision displacement of the player and of every chasing
  enemy, in screen px/s, so "can the player outrun this enemy type?" is a table
  instead of an assumption. Positions come from ``info.player_cell`` (the
  absolute cell published for the baseline) plus each slot's published cell
  delta, so the measurement needs no engine access;
* **attack attribution** — every enemy attack (a transition into ``windup``) is
  followed through its windup and active frames: was the player inside the
  enemy's *oriented* attack volume at the start of the windup and on the active
  frames, and did the player take damage in that window? Damage taken is also
  attributed to the state the enemy pack was in when it arrived.

::

    uv run python -m scripts.trace_episode --seed 2 --out /tmp/trace.csv
    uv run python -m scripts.trace_episode --seed 2 --summary
    uv run python -m scripts.trace_episode --seed 1 --seed-count 10 --summary

The summary aggregates the trace: decisions by branch (perceived / no target),
swings by enemy state-at-swing, swings by distance bucket, the true
``hits_landed / attacks_started`` land rate, the speed table and the attack
attribution table.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ashcrown_rl.baseline import DEFAULT_CONFIG, HeuristicPolicy
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.spaces import PX_PER_CELL, EnemySlot, Layout, screen_delta
from scripts.tune_baseline import VARIANTS

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
    "target_speed_px",
    "target_rel_x",
    "target_rel_y",
    "player_cell_x",
    "player_cell_y",
    "move",
    "attack",
    "interact",
    "started",
    "hits",
    "dealt",
    "taken",
)

#: Player hurtbox half-extents and vertical offset (``scenes/actors/player.tscn``).
PLAYER_HALF = (22.0, 28.0)
PLAYER_OFFSET_Y = -28.0
#: Legacy circle attack volume of the slimes (``enemy.tscn``: radius 24, centred
#: ``0.75 * attack_range`` in front of the feet).
LEGACY_CIRCLE_RADIUS_PX = 24.0
LEGACY_CIRCLE_FORWARD = 0.75
#: Documented body width of the oriented attack box: ``44 * scale_mult``.
BOX_WIDTH_PER_SCALE = 44.0


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


def _cell_of(value: Any) -> tuple[float, float] | None:
    """``info.player_cell`` as a finite ``(x, y)`` pair, or ``None``."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    pair = (float(value[0]), float(value[1]))
    return pair if all(np.isfinite(pair)) else None


def _screen_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Screen distance in px between two iso grid cells."""
    delta = (b[0] - a[0], b[1] - a[1])
    return float(np.hypot(*screen_delta(delta))) * PX_PER_CELL


# --------------------------------------------------------------------------- #
# Enemy identity tracking
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    """One enemy followed across decisions (slots are nearest-first, so the
    slot index is not an identity)."""

    key: int
    type_name: str
    cell: tuple[float, float]
    previous_cell: tuple[float, float]
    state: str
    state_progress: float
    last_seen: int
    attack: dict[str, Any] | None = None


@dataclass
class AttackEvent:
    """One enemy attack, from the windup start through its active frames."""

    type_name: str
    decision: int
    dist_windup_px: float
    reach_px: float
    box_width_px: float
    directional: bool
    in_box_windup: bool
    min_dist_px: float = 1e9
    dist_active_px: float | None = None
    in_box_active: bool | None = None
    lateral_margin_px: float = 0.0
    damage_taken: float = 0.0
    damage_while_swinging: float = 0.0
    target_died: bool = False
    closed: bool = False


def _inside_box(enemy: EnemySlot, rel_forward_px: tuple[float, float]) -> tuple[bool, float]:
    """Is the player inside this enemy's attack volume? Returns (hit, margin).

    ``rel_forward_px`` is the player's feet relative to the enemy's feet, in
    screen pixels. The oriented box runs from the enemy's feet to
    ``attack_reach_px`` along its facing with the documented body width; the
    player's hurtbox is the 44x56 rectangle centred 28 px above its feet. The
    margin is how far (px) the hurtbox would have to move perpendicular to the
    facing to leave the volume — the distance a strafe has to cover.
    """
    cos, sin = enemy.facing
    norm = float(np.hypot(cos, sin))
    if norm <= 1e-6:
        cos, sin = 1.0, 0.0
    else:
        cos, sin = cos / norm, sin / norm
    # Player hurtbox centre relative to the enemy's feet.
    px = rel_forward_px[0]
    py = rel_forward_px[1] + PLAYER_OFFSET_Y
    forward = px * cos + py * sin
    lateral = -px * sin + py * cos
    half_forward = PLAYER_HALF[0] * abs(cos) + PLAYER_HALF[1] * abs(sin)
    half_lateral = PLAYER_HALF[0] * abs(sin) + PLAYER_HALF[1] * abs(cos)
    directional = not enemy.type_name.startswith("slime") and enemy.attack_reach_px > 0.0
    if directional:
        width = BOX_WIDTH_PER_SCALE * enemy.scale_mult
        inside = (
            forward >= -half_forward
            and forward <= enemy.attack_reach_px + half_forward
            and abs(lateral) <= width * 0.5 + half_lateral
        )
        margin = width * 0.5 + half_lateral - abs(lateral)
        return inside, margin
    # Legacy circle volume: centred 0.75 * reach in front of the feet.
    centre = (
        cos * enemy.attack_reach_px * LEGACY_CIRCLE_FORWARD,
        sin * enemy.attack_reach_px * LEGACY_CIRCLE_FORWARD,
    )
    distance = float(np.hypot(px - centre[0], py - centre[1]))
    radius = LEGACY_CIRCLE_RADIUS_PX + max(PLAYER_HALF)
    return distance <= radius, radius - distance


def _px_between_cells(rel_cells: tuple[float, float]) -> tuple[float, float]:
    """The published cell delta as a screen px vector (enemy − player)."""
    screen = screen_delta(rel_cells)
    return (float(screen[0]) * PX_PER_CELL, float(screen[1]) * PX_PER_CELL)


def _match_tracks(
    tracks: dict[int, Track],
    enemies: list[EnemySlot],
    player_cell: tuple[float, float],
    decision: int,
) -> tuple[dict[int, tuple[Track, EnemySlot]], dict[int, Track]]:
    """Greedily pair this window's slots with the previous window's tracks."""
    matches: dict[int, tuple[Track, EnemySlot]] = {}
    used: set[int] = set()
    for enemy in sorted(enemies, key=lambda slot: slot.screen_distance_px):
        cell = (
            player_cell[0] + float(enemy.rel_cells[0]),
            player_cell[1] + float(enemy.rel_cells[1]),
        )
        best: tuple[float, int] | None = None
        for key, track in tracks.items():
            if key in used or track.type_name != enemy.type_name:
                continue
            gap = _screen_px(track.cell, cell)
            if gap > 3.0 * PX_PER_CELL:
                continue
            if best is None or gap < best[0]:
                best = (gap, key)
        if best is None:
            key = max(tracks, default=-1) + 1
            track = Track(
                key=key,
                type_name=enemy.type_name,
                cell=cell,
                previous_cell=cell,
                state=enemy.state,
                state_progress=enemy.state_progress,
                last_seen=decision,
            )
            tracks[key] = track
        else:
            key = best[1]
            track = tracks[key]
            track.previous_cell = track.cell
        matches[key] = (track, enemy)
        used.add(key)
        track.cell = cell
    return matches, tracks


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
def trace(
    seed: int,
    level: int,
    frame_skip: int,
    max_ticks: int,
    out: Path | None,
    *,
    quiet: bool = False,
    variant: str = "",
) -> dict[str, Any]:
    """Run one episode and return its measured tables (prints them unless quiet)."""
    env = AshCrownEnv(level=level, frame_skip=frame_skip, max_ticks=max_ticks)
    rows: list[dict[str, Any]] = []
    speed_samples: dict[str, list[float]] = {}
    player_speed: list[float] = []
    attacks: list[AttackEvent] = []
    damage_by_state: Counter[str] = Counter()
    tracks: dict[int, Track] = {}
    dt = float(frame_skip) / 60.0
    try:
        layout: Layout = env.layout
        obs, info = env.reset(seed=seed, options={"level": level})
        config = DEFAULT_CONFIG
        if variant:
            from dataclasses import replace

            config = replace(DEFAULT_CONFIG, **VARIANTS[variant])
        policy = HeuristicPolicy(layout, config)
        previous_cell = _cell_of(info.get("player_cell"))
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
                "target_cooldown_s": "" if target is None else round(target.attack_cooldown_s, 3),
                "target_reach_px": "" if target is None else round(target.attack_reach_px, 1),
                "target_hp_frac": "" if target is None else round(target.hp_frac, 3),
                "target_speed_px": "" if target is None else round(target.move_speed_px, 1),
                "target_rel_x": "" if target is None else round(target.rel_cells[0], 3),
                "target_rel_y": "" if target is None else round(target.rel_cells[1], 3),
                "player_cell_x": "" if previous_cell is None else round(previous_cell[0], 4),
                "player_cell_y": "" if previous_cell is None else round(previous_cell[1], 4),
                "move": int(action[0]),
                "attack": int(action[1]),
                "interact": int(action[2]),
            }
            obs, _reward, terminated, truncated, info = env.step(action)
            events = info.get("events") or {}
            taken = float(events.get("damage_taken", 0.0))
            row["started"] = int(events.get("attacks_started", 0))
            row["hits"] = int(events.get("hits_landed", 0))
            row["dealt"] = round(float(events.get("damage_dealt", 0.0)), 1)
            row["taken"] = round(taken, 1)
            rows.append(row)

            cell = _cell_of(info.get("player_cell"))
            if cell is not None and previous_cell is not None and player.state == "walk":
                walked = _screen_px(previous_cell, cell)
                if walked > 1.0:
                    player_speed.append(walked / dt)
            matches = _match_tracks(
                tracks,
                [slot for slot in layout.perceived_enemies(obs) if not slot.dead],
                cell if cell is not None else (0.0, 0.0),
                len(rows),
            )[0]
            _measure_speeds(matches, speed_samples, dt)
            _measure_attacks(layout, obs, matches, attacks, taken, damage_by_state, len(rows))
            previous_cell = cell
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
    tables = {
        "rows": rows,
        "player_speed": player_speed,
        "speed_samples": speed_samples,
        "attacks": attacks,
        "damage_by_state": damage_by_state,
        "layout": layout if rows else None,
    }
    if not quiet:
        _summarise(rows, player_speed, speed_samples, attacks, damage_by_state)
    return tables


def _measure_speeds(
    matches: dict[int, tuple[Track, EnemySlot]],
    speed_samples: dict[str, list[float]],
    dt: float,
) -> None:
    """Feed the per-type chase-speed samples from this window."""
    for track, enemy in matches.values():
        # ``track.state`` is the *previous* window's state: only a full
        # chase-to-chase interval measures the walk speed (a lunge, a knockback
        # or a windup would pollute it). The state is advanced by
        # :func:`_measure_attacks`, which runs after this.
        if enemy.state != "chase" or track.state != "chase":
            continue
        if enemy.asleep or enemy.screen_distance_px > 500.0:
            continue
        moved = _screen_px(track.previous_cell, track.cell)
        if moved > 0.5:
            speed_samples.setdefault(enemy.type_name, []).append(moved / dt)


def _measure_attacks(
    layout: Layout,
    obs: np.ndarray,
    matches: dict[int, tuple[Track, EnemySlot]],
    attacks: list[AttackEvent],
    damage_taken: float,
    damage_by_state: Counter[str],
    decision: int,
) -> None:
    """Track windup→active→recover per enemy and attribute the damage taken."""
    enemies = [slot for slot in layout.perceived_enemies(obs) if not slot.dead]
    states = [slot.state for slot in enemies]
    player = layout.player_features(obs)
    swinging = player.state == "attack"
    if any(slot.hitbox_active or slot.state == "attack" for slot in enemies):
        bucket = "attack"
    elif "windup" in states:
        bucket = "windup"
    elif "recover" in states:
        bucket = "recover"
    elif states:
        bucket = "chase/other"
    else:
        bucket = "none"
    damage_by_state[bucket] += damage_taken

    for track, enemy in matches.values():
        rel = _px_between_cells(enemy.rel_cells)
        # Player feet relative to the enemy's feet.
        rel_forward = (-rel[0], -rel[1])
        inside, margin = _inside_box(enemy, rel_forward)
        started = enemy.state == "windup" and track.state != "windup"
        if started:
            track.attack = AttackEvent(
                type_name=enemy.type_name,
                decision=decision,
                dist_windup_px=enemy.screen_distance_px,
                reach_px=enemy.attack_reach_px,
                box_width_px=BOX_WIDTH_PER_SCALE * enemy.scale_mult,
                directional=(
                    not enemy.type_name.startswith("slime") and enemy.attack_reach_px > 0.0
                ),
                in_box_windup=inside,
                min_dist_px=enemy.screen_distance_px,
            )
            attacks.append(track.attack)
        event = track.attack
        if event is not None and not event.closed:
            event.damage_taken += damage_taken
            if swinging:
                event.damage_while_swinging += damage_taken
            event.min_dist_px = min(event.min_dist_px, enemy.screen_distance_px)
            if (enemy.state == "attack" or enemy.hitbox_active) and event.dist_active_px is None:
                event.dist_active_px = enemy.screen_distance_px
                event.in_box_active = inside
                event.lateral_margin_px = margin
            if enemy.state in ("recover", "stagger", "idle"):
                event.closed = True
            if enemy.hp_frac <= 0.0:
                event.target_died = True
                event.closed = True
        track.state = enemy.state


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _summarise(
    rows: list[dict[str, Any]],
    player_speed: list[float],
    speed_samples: dict[str, list[float]],
    attacks: list[AttackEvent],
    damage_by_state: Counter[str],
) -> None:
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
    _speed_table(player_speed, speed_samples)
    _attack_table(attacks, damage_by_state, taken)


def _speed_table(player_speed: list[float], speed_samples: dict[str, list[float]]) -> None:
    """Observed per-second screen speed of the player and of each enemy type."""
    print("")
    print("speed while moving (screen px/s)   samples    mean     p95     max")
    if player_speed:
        values = np.asarray(player_speed)
        print(
            f"{'player (walk)':<32} {values.size:>8} "
            f"{values.mean():7.1f} {np.percentile(values, 95):7.1f} {values.max():7.1f}"
        )
    for name in sorted(speed_samples):
        values = np.asarray(speed_samples[name])
        if values.size == 0:
            continue
        print(
            f"{name + ' (chase)':<32} {values.size:>8} "
            f"{values.mean():7.1f} {np.percentile(values, 95):7.1f} {values.max():7.1f}"
        )


def _attack_table(
    attacks: list[AttackEvent],
    damage_by_state: Counter[str],
    total_taken: float,
) -> None:
    """Per-attack geometry and the state the damage actually arrived in."""
    print("")
    by_type: dict[str, list[AttackEvent]] = {}
    for event in attacks:
        by_type.setdefault(event.type_name, []).append(event)
    print("enemy attacks                n  in-box@windup  in-box@active  damage landed")
    for name in sorted(by_type):
        events = by_type[name]
        windup = sum(1 for event in events if event.in_box_windup)
        active = [event for event in events if event.in_box_active is not None]
        hit = sum(1 for event in active if event.in_box_active)
        damaged = sum(1 for event in events if event.damage_taken > 0.0)
        print(
            f"{name:<26} {len(events):>3} {windup / len(events):>13.0%} "
            f"{hit / max(len(active), 1):>13.0%} {damaged / len(events):>13.0%}"
        )
    landed = [event for event in attacks if event.damage_taken > 0.0]
    if landed:
        locked = sum(event.damage_while_swinging for event in landed)
        total = sum(event.damage_taken for event in landed)
        print(
            f"of the {len(landed)} attacks that landed, "
            f"{locked / total:.0%} of the damage arrived while the player body was "
            "locked in its own swing"
        )
    acted = [event for event in attacks if event.in_box_active is not None]
    if acted:
        margins = np.asarray([event.lateral_margin_px for event in acted])
        print(
            f"active-frame lateral margin: median {np.median(margins):.1f} px "
            f"(a perpendicular step of that size leaves the box)"
        )
    print("")
    print("damage taken attributed to the pack state at that decision")
    for key, value in sorted(damage_by_state.items(), key=lambda item: -item[1]):
        share = value / total_taken if total_taken else 0.0
        print(f"  {key:<12} {value:7.1f} ({share:5.0%})")


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
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--variant",
        default="",
        help="a scripts.tune_baseline.VARIANTS name to run instead of the shipped config",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="aggregate the seeds into one printed table instead of one block per seed",
    )
    args = parser.parse_args()
    seeds = list(range(args.seed, args.seed + max(args.seed_count, 1)))
    player_speed: list[float] = []
    speed_samples: dict[str, list[float]] = {}
    attacks: list[AttackEvent] = []
    damage_by_state: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        tables = trace(
            seed=seed,
            level=args.level,
            frame_skip=args.frame_skip,
            max_ticks=args.max_ticks,
            out=Path(args.out) if args.out and len(seeds) == 1 else None,
            quiet=args.summary and len(seeds) > 1,
            variant=args.variant,
        )
        if args.summary and len(seeds) > 1:
            print(f"--- seed {seed}")
            _summarise(
                tables["rows"],
                tables["player_speed"],
                tables["speed_samples"],
                tables["attacks"],
                tables["damage_by_state"],
            )
        player_speed.extend(tables["player_speed"])
        for name, values in tables["speed_samples"].items():
            speed_samples.setdefault(name, []).extend(values)
        attacks.extend(tables["attacks"])
        damage_by_state.update(tables["damage_by_state"])
        rows.extend(tables["rows"])
    if len(seeds) > 1:
        print("")
        print(f"=== aggregate over seeds {seeds[0]}..{seeds[-1]} ({len(rows)} decisions)")
        _speed_table(player_speed, speed_samples)
        total_taken = sum(float(row["taken"] or 0.0) for row in rows)
        _attack_table(attacks, damage_by_state, total_taken)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
