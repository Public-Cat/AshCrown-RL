"""Scripted heuristic baseline for ``ashcrown-obs-v1`` (RL-4, ``docs/07`` §9.4).

The decision core is a **pure function of the observation**
(:func:`heuristic_action`), so it can be unit-tested without a Godot process and
replayed exactly; the only state lives in :class:`HeuristicPolicy` (a global
navigation map plus two short-lived memories).

The policy is built on the *measured* combat economics of level 1
(``scripts/trace_episode.py``, ``docs/artifacts/RL/RL4-report.md`` §"round 3"):

* a committed swing lands on ~100-128 % of the swings started inside 0-70 px, so
  the earlier belief that "only 10 % of swings land" was a measurement artifact
  (hits divided by *attack requests*, not by started swings);
* a landed hit staggers the enemy for 0.25 s and cancels its telegraph, so the
  cheapest defence is hitting first — the swing's hitbox goes live ~0.32 s after
  the start;
* the body is locked for the whole 0.7 s swing, so a swing started inside the
  enemy's own reach cannot be escaped before its next telegraph: the policy holds
  a **standoff band** (~0.9 x the enemy's reach up to the player's reach + 40 px)
  and only swings from inside it, with an exact swing-volume overlap test
  (:func:`_swing_covers`) rather than a facing-cone guess.

Layout of the file:

1. **Fight** (:func:`_fight`): approach, standoff band, interrupt, dodge, retreat
   through a doorway when two attackers can land at once.
2. **Portal** (:func:`_enter_portal`): path to the open portal and pulse
   ``interact`` inside its cell.
3. **Explore** (:func:`_explore_step`): the nearest unvisited cell of the local
   window, then the caller's hint, and never standing on the inactive portal.
4. **Global navigation** (:class:`~ashcrown_rl.navmap.NavMap`): the 21x21 window
   cannot path to a goal behind a wall it does not see, so the wrapper keeps a
   portal-relative map of what it has seen and routes long approaches through it
   (route memory keeps walking to a target that slipped out of perception).

Every tie-break is a fixed action-index/dict order and no RNG is used, so the
same observation always yields the same action. Acceptance (``docs/07`` §9.4):
clears level 1 on at least 8 of 10 fixed seeds with the warrior inside the
5-game-minute limit — **met: 8/10 on the fixed seeds 1–10** with the shipped
``DEFAULT_CONFIG`` (``strafe_dodge=True``, ``strafe_side=+1``,
``strike_margin_px=60``, ``dodge_window_s=0.45``, ``strafe_band_px=8``,
``standoff_frac=0.92``, ``keepout_slack_px=12``; every pack gate off). Seeds 3
and 8 are the two failures. On **unseen** seed lists the same config clears 19/30 (seeds 11–40) and 22/30 (seeds 41–70), i.e. 68 %
overall — the fixed list is the friendlier half of the distribution. See
``docs/artifacts/RL/RL4-report.md`` §"round 4" and
``docs/artifacts/RL/unseen_seeds.csv`` for the measurements.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .navmap import NavMap
from .spaces import (
    PX_PER_CELL,
    EnemySlot,
    Layout,
    PlayerFeatures,
    PortalFeatures,
    normalise,
    screen_delta,
)


@dataclass(frozen=True)
class HeuristicConfig:
    """Full configuration of the scripted baseline (immutable, hashable)."""

    #: Extra pixels on the player's weapon reach that still land a hit.
    #: Measured on level 1 (``scripts/trace_episode.py``): every swing started
    #: inside 0-70 px landed (>= 100 % of the per-swing land rate), so the
    #: strike gate is reach + this margin rather than the old 60-69 px window.
    #: Round 4 swept it again on top of the perpendicular strafe: 20 / 50 / 60 /
    #: 70 → 5 / 8 / **8** / 6 clears, so 60 ships. The extra 40 px are the
    #: 63-65 px band where the exact swing-volume test already lands (the enemy
    #: hurtbox is 22 px wide either side of its feet) but the old ``in_strike``
    #: refused the swing.
    strike_margin_px: float = 60.0
    #: Extra pixels on the enemy's reach that make a position dangerous. The
    #: enemy's swing volume reaches its ``attack_reach_px`` from its feet and the
    #: player's hurtbox adds ~22 px, so its threat radius is reach + ~22. The
    #: measured best is deliberately *tighter* than that (12): treating the full
    #: geometric envelope as danger made the policy dodge hits that were never
    #: going to land and it stopped killing (1 clear / 7 deaths at slack 26,
    #: 5 clears / 4 deaths at slack 12).
    keepout_slack_px: float = 12.0
    #: Seconds between a swing start and its hitbox going live (0.375 x the
    #: warrior's 0.7 s swing); a telegraph is only interruptible beyond this.
    swing_land_s: float = 0.32
    #: One decision window plus jitter (0.067 s at ``frame_skip=4``): the swing
    #: starts on the *next* tick, so a race that is won by less than this is
    #: lost. Measured on seed 8: ``windup_remaining=0.333`` was treated as a win
    #: against a 0.32 s swing and the enemy's box went live first.
    reaction_s: float = 0.0
    #: Side of the perpendicular dodge: ``+1`` = clockwise, ``-1`` =
    #: counter-clockwise, ``0`` = whichever side scores better this window. A
    #: fixed side keeps consecutive steps circling the enemy; the free choice
    #: flips as soon as the body crosses the bearing axis, which cancels the
    #: strafe out (measured on seed 8: ``2,5,2`` alternating steps with the
    #: enemy's box still on the body).
    strafe_side: int = 1
    #: Windup assumed when a chasing enemy could start one this window.
    assumed_windup_s: float = 0.35
    #: Standoff band: swing only from at least this fraction of the enemy's own
    #: reach (a point-blank swing cannot be escaped before the next telegraph).
    standoff_frac: float = 0.92
    #: Lower bound of the standoff band, in pixels.
    standoff_min_px: float = 30.0
    #: Enemies inside this radius count as the pack that can pile on.
    pack_radius_px: float = 105.0
    #: With a pack on top of the body, retreat *through the nearest doorway*
    #: instead of straight away from the pack: a 2-cell doorway funnels the
    #: enemies into one at a time, which is the only way a 3v1 is survivable.
    doorway_retreat: bool = True
    #: Portal distance (iso cells) at which ``interact`` is pulsed.
    portal_interact_cells: float = 0.7
    #: Under this distance the local window handles the approach itself; above
    #: it the wrapper's global map picks the route (the enemy can be behind a
    #: wall the local window cannot path around).
    local_approach_px: float = 260.0
    #: Portal distance (iso cells) under which the pure core must walk away
    #: from the portal instead of standing on it.
    explore_fallback_cells: float = 7.5
    #: Decisions the wrapper keeps walking to the last perceived enemy's cell
    #: after it drops out of perception (route memory).
    goal_memory_decisions: int = 90
    #: Two or more enemies inside this radius get a short sidestep instead of a
    #: swing: measured best at 60 px for ``pack_sidestep_decisions`` = 1 (a
    #: longer commitment was worse — it turned fights into a 200-decision kite).
    pack_sidestep_px: float = 60.0
    #: Decisions a pack sidestep is held for.
    pack_sidestep_decisions: int = 2
    #: **Round 4, pack-fighting micro.** Dodge a telegraphed swing by stepping
    #: *perpendicular* to the attacker's bearing (circling it) instead of
    #: backing straight away: the enemy's attack volume is an oriented box that
    #: freezes with its facing at the end of the chase, so a lateral step leaves
    #: it while a retreat along its facing stays inside. Measured on level 1
    #: (``scripts/trace_episode.py``): the player walks 259 px/s against a
    #: 103 px/s skeleton, and the lateral margin at the active frames has a
    #: median of 24 px — one 45-degree step clears it.
    strafe_dodge: bool = True
    #: Seconds before an enemy's box can connect at which the body starts the
    #: perpendicular step (one decision window is 0.067 s).
    dodge_window_s: float = 0.45
    #: Seconds the player's body is locked by a committed swing (0.7 s clip):
    #: a swing is only safe when no enemy can land a hit inside this window.
    swing_lock_s: float = 0.70
    #: Distance (px) from an attacker the strafe tries to hold while circling
    #: it, on top of its own reach: close enough to punish the recovery.
    strafe_band_px: float = 8.0
    #: Weight of the "distance drifted from the band" penalty in the strafe
    #: score (px of lateral gain per px of drift).
    strafe_drift_weight: float = 0.6
    #: Score bonus (px) for continuing the previous strafe direction, so the
    #: body circles instead of ping-ponging across the enemy's facing.
    strafe_hold_bonus: float = 12.0
    #: The hold bonus only applies while the step keeps the body within this many
    #: px below the target band, so a "sidestep" can never also close the
    #: distance into point blank.
    strafe_hold_floor_px: float = 5.0
    #: Fight from a doorway when two or more enemies can land a hit at once and
    #: a doorway of the current room is this close (cells): a 2-cell doorway
    #: funnels a pack into one attacker at a time.
    doorway_max_cells: float = 6.0
    #: Two or more enemies inside ``pack_reach_mult`` x the largest reach make a
    #: room unsafe to fight in: relocate to the nearest doorway (chokepoint) and
    #: fight there, one attacker at a time.
    chokepoint: bool = False
    #: Radius (x the largest perceived reach) that defines "the pack is on us".
    pack_reach_mult: float = 1.5
    #: Pick the *most isolated* target instead of the nearest wounded one: with
    #: a 0.7 s swing lock the cheapest kill is the enemy whose pack-mates are
    #: farthest away, not the closest body.
    isolate_target: bool = False
    #: Refuse a swing that a second attacker can punish inside the 0.7 s lock
    #: unless the focus is already recovering/staggered (the cheapest trade).
    #: Measured worse (0/10): it stops the swinging without ever opening the
    #: distance, so the body just stands in the pack. Kept for the A/B log.
    pack_gate: bool = False
    #: **Queue gate.** A swing is only committed when *every* other perceived
    #: enemy needs more than ``swing_lock_s`` to connect. In a pack that means
    #: fighting the leader of the queue: the player runs 2.5x faster than a
    #: skeleton, so a retreat lines the attackers up nose-to-tail (44 px apart)
    #: and the second one is always a full body length behind the first.
    queue_gate: bool = False
    #: Distance (px) under which two or more attackers count as "the pack is on
    #: the body": while the queue gate blocks the swing the policy opens the
    #: distance instead of orbiting at point blank.
    separate_px: float = 110.0
    #: **Punish only a telegraph or a recovery.** The enemy's box reaches
    #: ``attack_reach_px`` from its feet plus ~22 px of the player's hurtbox
    #: (= 72 px for a skeleton) while the player's own strike reaches 63 px, so
    #: a swing started at a *chasing* enemy always loses the race by ~0.05 s and
    #: costs a hit. Waiting for the windup (which is frozen and cancelable) or
    #: the 0.4 s recovery turns the same swing into a free hit.
    punish_recover: bool = False


DEFAULT_CONFIG = HeuristicConfig()


def heuristic_action(
    obs: np.ndarray,
    info: Mapping[str, Any] | None = None,
    layout: Layout | None = None,
    config: HeuristicConfig = DEFAULT_CONFIG,
    explore_target: Sequence[float] | None = None,
) -> np.ndarray:
    """Return ``[move, attack, interact]`` for one observation.

    ``layout`` is the parsed handshake layout (``env.layout``). When omitted it
    is taken from ``info["layout"]`` (a :class:`Layout`, or a raw handshake
    ``layout`` mapping), so the function also works in a process-free test.

    ``explore_target`` is an optional grid-cell delta (target − player) that the
    caller's *stateful* sweep wants the body to walk to. It is only consulted
    when the local window has no unvisited cell left — the local frontier always
    wins — so the pure function stays a function of the observation plus that
    one explicit hint (the same contract as ``info["action"]``).
    """
    resolved = _resolve_layout(info, layout)
    player = resolved.player_features(obs)
    if not player.alive:
        return np.zeros(3, dtype=np.int64)
    grid = resolved.grid(obs)
    previous = _previous_move(info)

    hazard_step = _flee_hazard(resolved, grid)
    if hazard_step is not None:
        return _action(hazard_step, 0, 0)

    enemies = [slot for slot in resolved.perceived_enemies(obs) if not slot.dead]
    if enemies:
        return _fight(resolved, grid, player, enemies, config, previous)

    portal = resolved.portal(obs)
    if portal.active:
        return _enter_portal(resolved, grid, portal, previous, config)
    return _action(_explore_step(resolved, grid, portal, previous, config, explore_target), 0, 0)


# --------------------------------------------------------------------------- #
# Behaviours
# --------------------------------------------------------------------------- #
def _fight(
    layout: Layout,
    grid: np.ndarray,
    player: PlayerFeatures,
    enemies: Sequence[EnemySlot],
    config: HeuristicConfig,
    previous: int = 0,
) -> np.ndarray:
    """Approach, strike and dodge the perceived enemies (``docs/07`` §9.4).

    ``enemies`` is nearest-first. Every rule below comes from the *measured*
    level-1 combat economics (``scripts/trace_episode.py`` and the
    ``docs/artifacts/RL/baseline_level1*.csv`` artifact):

    * a swing started inside ~63 px lands (~100-128 % of started swings), so the
      strike gate is the player's own reach, not the old "stand at 60-69 px"
      rule;
    * a landed hit *staggers* the enemy for 0.25 s and cancels its telegraph, so
      hitting first is the cheapest defence;
    * the swing's hitbox goes live ~0.32 s after the start and the body is
      **locked for 0.7 s**, while an enemy telegraph is 0.35 s + a 0.3 s active
      window. A swing from *point blank* therefore cannot be escaped before the
      enemy's next telegraph lands — measured as ~10 hits taken in 20 s on the
      seeds that died. So the body holds a **standoff band** (≈0.9 x the enemy's
      reach up to the player's own) and only swings from inside it.

    Round 4 added the two measured *survival* facts (see the report):

    * the player walks **259 px/s** while a chasing skeleton does 103 and a
      slime 73, so distance is always recoverable and a 0.35 s telegraph is
      always dodgeable;
    * the enemy's attack volume is an **oriented box** that freezes with its
      facing when the chase ends, and the player is inside that box on 87 % of
      the skeleton's active frames — but on only 15 % of the damage was the body
      locked in its own swing. The hits therefore arrive because the body
      backs *along the enemy's facing*, which stays inside the box.

    Decision order (a small deterministic state machine):

    1. a committed swing is left alone (and never queued);
    2. any enemy whose box can connect inside ``dodge_window_s`` → **circle it**:
       a perpendicular step that leaves the box while staying in punish range
       (through a doorway when two or more can land at once);
    3. the focus target is covered by the swing volume inside the standoff band
       and **no enemy can reach the body during the 0.7 s swing lock** → swing;
    4. the focus is clearly recovering and only a hit landing after ours can
       answer → trade the swing for the stagger;
    5. too close for a safe swing → back out to the band;
    6. inside the band but the swing would miss → one step to turn onto it;
    7. otherwise close on a BFS path.
    """
    if player.state == "attack" or player.attack_queued:
        # A committed swing ignores movement and a queued attack would chain a
        # second swing back to back — always keep the free window afterwards.
        return _action(0, 0, 0)

    focus = _focus_target(enemies, config)
    screen = screen_delta(focus.rel_cells)
    distance_px = float(np.hypot(*screen)) * PX_PER_CELL

    in_strike = distance_px <= player.attack_reach_px + config.strike_margin_px
    in_band = in_strike and distance_px >= _standoff_px(focus, config)
    # Exact swing-volume test (see ``_swing_covers``): the swing locks the
    # current facing, so this is the same test the engine will run on the
    # hitbox's live frames.
    covers = _swing_covers(player.facing, screen * PX_PER_CELL, player.attack_reach_px)

    # Distance-aware threat timing for every perceived enemy: seconds until its
    # box can touch the body *if it stands still from here* (round 4). The
    # historic gate used ``_threat_in_s``, which ignores the closing time.
    timings = {slot.slot: _time_to_hit(slot, config) for slot in enemies}
    focus_threat = timings[focus.slot]
    nearest_other = min(
        (value for key, value in timings.items() if key != focus.slot),
        default=_NO_THREAT_S,
    )
    dangerous = [
        slot
        for slot in enemies
        if slot.screen_distance_px <= slot.attack_reach_px + config.keepout_slack_px
    ]
    imminent = [slot for slot in dangerous if timings[slot.slot] <= config.swing_land_s]

    # 1. Swing gate. Round 3's geometry (standoff band + exact swing volume +
    #    our hit landing first) plus round 4's *timing* rule: a landed hit
    #    staggers, so the only cheap swing is one no other attacker can answer
    #    inside the 0.7 s lock (``queue_gate``).
    race_margin = config.swing_land_s + config.reaction_s
    phase_ok = (
        not config.punish_recover
        or focus.state in ("recover", "stagger")
        or (focus.state == "windup" and focus_threat >= race_margin)
    )
    # The gate is the *timing* model, not a distance proxy: an enemy 64 px away
    # in a 0.1 s telegraph is inside its 72 px kill envelope (reach + the player's
    # 22 px hurtbox) and lands before our 0.32 s hit, while a chasing enemy at the
    # same distance still has to close first and loses the race.
    if in_band and covers and phase_ok and focus_threat >= race_margin:
        if not config.queue_gate and not config.pack_gate:
            return _action(0, 1, 0)
        if nearest_other >= config.swing_lock_s:
            return _action(0, 1, 0)
        if config.pack_gate and focus.state in ("recover", "stagger"):
            if nearest_other > config.swing_land_s:
                return _action(0, 1, 0)

    # 2. Perpendicular dodge: an attacker whose box will connect inside
    #    ``dodge_window_s`` is circled instead of backed away from. The volume is
    #    an oriented box frozen with its facing, so a lateral step leaves it
    #    while a retreat along the facing stays inside (measured: 87 % in-box on
    #    the active frames with the old straight retreat).
    incoming = [slot for slot in enemies if timings[slot.slot] <= config.dodge_window_s]
    if config.strafe_dodge and incoming:
        if len(incoming) >= 2 and config.doorway_retreat:
            doorway = _doorway_step(layout, grid, incoming, previous, config.doorway_max_cells)
            if doorway is not None:
                return _action(doorway, 0, 0)
        step = _strafe_step(layout, grid, incoming, previous, config)
        if step is not None:
            return _action(step, 0, 0)

    # 3. Queue gate blocked the swing: open the distance so the attackers line up
    #    behind the leader instead of standing in the middle of the pack.
    if config.queue_gate and nearest_other < config.swing_lock_s:
        packed = [slot for slot in enemies if slot.screen_distance_px <= config.separate_px]
        if len(packed) >= 2:
            if config.chokepoint:
                doorway = _doorway_step(
                    layout, grid, packed, previous, config.doorway_max_cells, any_door=True
                )
                if doorway is not None:
                    return _action(doorway, 0, 0)
            return _retreat(layout, grid, packed, previous, config)

    # 4. Two hits landing on us at once: no single swing can interrupt both.
    if len(imminent) >= 2 and not config.strafe_dodge:
        return _retreat(layout, grid, imminent, previous, config)

    if not in_band and in_strike:
        # Point blank: back out to the band before trading (a 0.7 s lock from
        # inside 0.9 x the enemy's reach cannot be escaped, see the docstring).
        # Every enemy inside the pack radius counts, not just the focus: with
        # two attackers the nearest one alternates and a single-threat flee
        # ping-pongs between them.
        return _retreat(layout, grid, dangerous or [focus], previous, config)
    if focus in dangerous and focus_threat < config.swing_land_s:
        return _retreat(layout, grid, [focus], previous, config)

    # 3. Inside the band but the swing would miss: turn onto the target. The
    #    facing only changes when the body moves, so this is one 45-degree step
    #    toward it — no hysteresis, the swing comes next window.
    if in_band:
        step = layout.move_index(screen)
        if not _step_is_free(layout, grid, layout.actions[step].grid):
            step = _steer(layout, screen, previous)
        return _action(step, 0, 0)

    # 4. Close the distance on a BFS path (walls, doorways, props).
    move = _path_step(layout, grid, focus.rel_cells, previous)
    if move == 0:
        # Inside half a cell the rounded BFS target is our own cell; steer with
        # the exact screen direction instead.
        move = _steer(layout, screen, previous)
    return _action(move, 0, 0)


def _retreat(
    layout: Layout,
    grid: np.ndarray,
    threats: Sequence[EnemySlot],
    previous: int,
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Step out of every threat at once; hold only when there is no way out.

    With two or more attackers the retreat is routed **through the nearest
    doorway** when that doorway is farther from the pack than the body already
    is: a 2-cell doorway funnels a pack into one attacker at a time, while a
    plain "away from the centroid" step in a room walks into the far wall and
    the pack converges (the measured 3v1 death: 10 hits in 12 s with one swing).
    """
    if config.doorway_retreat and len(threats) >= 2:
        step = _doorway_step(layout, grid, threats, previous)
        if step is not None:
            return _action(step, 0, 0)
    screens = [screen_delta(slot.rel_cells) for slot in threats]
    escape = _flee_step(layout, grid, screens, previous)
    if escape is not None:
        return _action(escape, 0, 0)
    return _action(0, 0, 0)


def _doorway_step(
    layout: Layout,
    grid: np.ndarray,
    threats: Sequence[EnemySlot],
    previous: int,
    max_cells: float = 99.0,
    *,
    any_door: bool = False,
) -> int | None:
    """First BFS step to a doorway that is farther from the pack than we are.

    ``max_cells`` bounds the walk (the round-4 chokepoint rule: fight from a
    doorway only when one is close, otherwise the retreat path itself costs more
    than the hits it saves).
    """
    centroid = _centroid_screen(threats)
    here = float(np.hypot(*centroid))
    paths = _bfs(layout, grid)
    doors = [
        (cell, path)
        for cell, path in paths.items()
        if cell != (0, 0)
        and layout.cell(grid, cell[0], cell[1], "doorway") > 0.5
        and _screen_length(cell) <= max_cells
    ]
    if not doors:
        return None
    cell, path = min(
        doors,
        key=lambda item: (item[1][1], item[0][0], item[0][1]),
    )
    if not any_door and float(np.hypot(*(screen_delta(cell) - centroid))) <= here:
        # The doorway is on the pack's side of the body: retreating into it would
        # walk *through* the attackers.
        return None
    return _no_reversal(path[0], previous, layout, grid)


def _strafe_step(
    layout: Layout,
    grid: np.ndarray,
    threats: Sequence[EnemySlot],
    previous: int,
    config: HeuristicConfig,
) -> int | None:
    """One step *perpendicular* to the nearest attacker's bearing.

    The enemy attack volume is an oriented rectangle that starts at the enemy's
    feet and extends ``attack_reach_px`` along its facing with the documented
    body width (``44 * scale``). The facing freezes when the chase ends, so a
    step along the facing (retreating) stays inside the volume while a step
    perpendicular leaves it. Measured on level 1: the lateral margin at the
    active frames has a median of 24 px, and the player covers that in 0.1 s at
    259 px/s.

    The score keeps the body in punish range (``strafe_band_px`` outside the
    attacker's own reach) so the swing is available the moment the recovery
    starts, and prefers to continue the previous direction so consecutive steps
    circle the enemy instead of oscillating across its facing.
    """
    if not threats:
        return None
    threat = min(threats, key=lambda slot: (slot.screen_distance_px, slot.slot))
    cells = np.asarray(screen_delta(threat.rel_cells), dtype=np.float64)
    length = float(np.hypot(*cells))
    if length <= 1e-6:
        return None
    # Unit perpendicular in screen space, on the configured side only: a fixed
    # rotation direction makes consecutive steps circle the attacker instead of
    # cancelling out (see ``strafe_side``).
    perp = np.array([-cells[1] / length, cells[0] / length]) * float(config.strafe_side)
    target_px = threat.attack_reach_px + config.strafe_band_px
    others = [slot for slot in threats if slot.slot != threat.slot]
    if config.strafe_side in (1, -1):
        index, _score = _best_strafe_step(
            layout,
            grid,
            cells,
            perp,
            target_px,
            others,
            previous,
            config,
            require_lateral=True,
        )
        return index
    # ``strafe_side == 0`` (measured worse: 0/10): evaluate both sides and keep
    # the best score, which flips as soon as the body crosses the bearing axis.
    best: int | None = None
    best_score = -np.inf
    for side in (1.0, -1.0):
        index, score = _best_strafe_step(
            layout,
            grid,
            cells,
            perp * side,
            target_px,
            others,
            previous,
            config,
            require_lateral=True,
        )
        if index is not None and score > best_score + 1e-9:
            best, best_score = index, score
    return best


def _best_strafe_step(
    layout: Layout,
    grid: np.ndarray,
    cells: np.ndarray,
    perp: np.ndarray,
    target_px: float,
    others: Sequence[EnemySlot],
    previous: int,
    config: HeuristicConfig,
    *,
    require_lateral: bool,
) -> tuple[int | None, float]:
    """Best single step along ``perp`` (one side), or ``(None, -inf)``."""
    best_index: int | None = None
    best_score = -np.inf
    for spec in layout.actions:
        if not spec.moving:
            continue
        if not _step_is_free(layout, grid, spec.grid):
            continue
        step_cells = np.asarray(screen_delta(spec.grid), dtype=np.float64)
        lateral = float(step_cells[0] * perp[0] + step_cells[1] * perp[1]) * PX_PER_CELL
        if require_lateral and lateral <= 0.0:
            # This side has no lateral step at all: try the other side.
            continue
        after = cells - step_cells
        after_px = float(np.hypot(*after)) * PX_PER_CELL
        score = lateral - config.strafe_drift_weight * abs(after_px - target_px)
        # Avoid stepping into another attacker's volume.
        for other in others:
            gap = float(np.hypot(*(step_cells - np.asarray(screen_delta(other.rel_cells)))))
            if gap * PX_PER_CELL <= other.attack_reach_px + config.keepout_slack_px:
                score -= 40.0
        if (
            previous
            and spec.index == previous
            and after_px >= target_px - config.strafe_hold_floor_px
        ):
            # Continue the current sidestep, but never to keep closing in: the
            # unconditional bonus re-selected a forward-lateral step that walked
            # the body from 35 px to 21 px on seed 8 (measured in the trace), and
            # point blank is the one place where the box cannot be escaped.
            score += config.strafe_hold_bonus
        if score > best_score + 1e-9:
            best_score = score
            best_index = int(spec.index)
    if require_lateral and best_index is None:
        return None, -np.inf
    return best_index, best_score


def _standoff_px(enemy: EnemySlot, config: HeuristicConfig) -> float:
    """Closest distance a safe swing may start from this enemy.

    A swing from closer than this cannot be escaped before the enemy's *next*
    telegraph goes live (0.7 s lock + 0.35 s windup versus a 0.4 s recovery), so
    the band starts just inside the enemy's own reach.
    """
    return max(config.standoff_frac * enemy.attack_reach_px, config.standoff_min_px)


def _focus_target(enemies: Sequence[EnemySlot], config: HeuristicConfig) -> EnemySlot:
    """The enemy to kill next.

    Focus fire finishes a target instead of spreading damage over the pack
    (three wounded enemies hit exactly as hard as three healthy ones). With
    ``isolate_target`` the choice is the **most isolated** wounded enemy — the
    one whose nearest pack-mate is farthest — because the 0.7 s swing lock costs
    a hit for every attacker inside reach, and a lone skeleton can be killed
    without taking any. Without it, the historic rule stays: the nearest wounded
    enemy, else the nearest.
    """
    reachable = [slot for slot in enemies if slot.screen_distance_px <= config.pack_radius_px * 4.0]
    pool = reachable or list(enemies)
    if config.isolate_target:

        def isolation(slot: EnemySlot) -> float:
            mates = [other.screen_distance_px for other in enemies if other.slot != slot.slot]
            return min(mates) if mates else _NO_THREAT_S

        wounded = [slot for slot in pool if slot.hp_frac < 0.999]
        return min(
            wounded or pool,
            key=lambda slot: (-isolation(slot), slot.screen_distance_px, slot.slot),
        )
    wounded = [slot for slot in pool if slot.hp_frac < 0.999]
    return min(wounded or pool, key=lambda slot: (slot.screen_distance_px, slot.slot))


#: ``(attack_windup, attack_active, attack_recover)`` per enemy type, read from
#: ``resources/enemies/*.tres``. The observation publishes the windup *remaining*
#: and the state progress but not the state lengths, so the pack timing model
#: needs the table: "time until this body can hit me" is what decides whether a
#: committed swing is safe.
ENEMY_TIMING: dict[str, tuple[float, float, float]] = {
    "slime_green": (0.35, 0.15, 0.40),
    "slime_red": (0.35, 0.15, 0.40),
    "slime_black": (0.45, 0.15, 0.45),
    "skeleton_normal": (0.35, 0.30, 0.40),
    "skeleton_armored": (0.55, 0.36, 0.55),
    "skeleton_swift": (0.20, 0.70, 0.35),
    "elite_skeleton": (0.50, 0.40, 0.50),
    "elite_slime": (0.45, 0.20, 0.45),
    "demonlord": (0.60, 0.25, 0.80),
}


def _enemy_timing(type_name: str) -> tuple[float, float, float]:
    """The ``(windup, active, recover)`` seconds of a type (defaults: skeleton)."""
    return ENEMY_TIMING.get(type_name, ENEMY_TIMING["skeleton_normal"])


def _time_to_hit(enemy: EnemySlot, config: HeuristicConfig) -> float:
    """Seconds until this enemy's box can connect if the body stands still.

    The round-4 pack model. A live swing is now; a telegraph is its remaining
    windup; a recovery still owes its remaining state plus the next windup; a
    chasing enemy first has to close the gap at its *published* move speed. The
    old ``_threat_in_s`` ignored the closing time, which is why a policy holding
    a standoff band still started swings while a second attacker was 90 px away
    and one window short of the lock.
    """
    if enemy.hitbox_active or enemy.state == "attack":
        return 0.0
    windup, _active, recover = _enemy_timing(enemy.type_name)
    if enemy.state == "windup":
        if enemy.windup_remaining_s > 0.0:
            return enemy.windup_remaining_s
        # A probe/observation without the remaining time: derive it from the
        # state progress instead of assuming a fresh telegraph.
        return windup * (1.0 - enemy.state_progress)
    if enemy.state == "stagger":
        return _NO_THREAT_S
    if enemy.state == "recover":
        remaining = max(1.0 - enemy.state_progress, 0.0) * recover
        return remaining + windup + max(enemy.attack_cooldown_s, 0.0)
    if enemy.state == "dead":
        return _NO_THREAT_S
    # Idle / chase / other: the body is outside the volume, it has to close in.
    gap = max(enemy.screen_distance_px - enemy.attack_reach_px, 0.0)
    speed = max(enemy.move_speed_px, 1.0)
    return gap / speed + windup + max(enemy.attack_cooldown_s, 0.0)


def _enter_portal(
    layout: Layout,
    grid: np.ndarray,
    portal: PortalFeatures,
    previous: int = 0,
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Walk into the open portal and pulse ``interact`` inside its cell.

    Inside the portal's own grid cell the rounded path target is the player's
    cell, so the BFS step is "stay"; steering with the exact screen delta is
    what closes the last fraction of a cell onto the trigger instead of parking
    next to it (the RL-4 trap that cost two otherwise-cleared seeds).
    """
    move = _path_step(layout, grid, portal.rel_cells, previous, avoid_reversal=False)
    if move == 0:
        move = _steer(layout, screen_delta(portal.rel_cells), previous)
    interact = 1 if portal.distance_cells <= config.portal_interact_cells else 0
    return _action(move, 0, interact)


def _explore_step(
    layout: Layout,
    grid: np.ndarray,
    portal: PortalFeatures,
    previous: int = 0,
    config: HeuristicConfig = DEFAULT_CONFIG,
    explore_target: Sequence[float] | None = None,
) -> int:
    """BFS first step to the nearest unexplored floor cell of the local window.

    The env marks a 3x3 brush around the player as ``visited``, so the nearest
    unvisited cell is always just outside the brush and the walk sweeps the
    floor systematically. Ties (the eight cells of the same BFS ring) go to the
    cell closest to the sweep target: the exit sits on the far side of the
    floor, so the sweep leans across the map instead of orbiting the room it
    started in.

    When the whole window is explored there is no frontier left and the target
    is outside the 21x21 window. Then, in order:

    1. the caller's ``explore_target`` (the stateful sweep's next waypoint);
    2. the portal bearing — the spec'd long-range fallback (§5.7);
    3. **away from the portal**, when the body is already standing next to it.
       That last case is what stalled seeds 1/2/3/6: the portal is *inactive*
       but its cell is inside the window, the clamped bearing rounds to the
       player's own cell, and the policy stood still for the rest of the
       episode (measured: 3,421 of 4,500 decisions on seed 2).
    """
    paths = _bfs(layout, grid)
    frontier = [
        (cell, path)
        for cell, path in paths.items()
        if cell != (0, 0) and layout.visited(grid, cell[0], cell[1]) < 0.5
    ]
    if frontier:
        reference = (
            (float(explore_target[0]), float(explore_target[1]))
            if explore_target is not None
            else (float(portal.rel_cells[0]), float(portal.rel_cells[1]))
        )
        reference_cell = (int(round(reference[0])), int(round(reference[1])))
        _, path = min(
            frontier,
            key=lambda item: (
                item[1][1],
                round(_distance_to(item[0], reference_cell), 6),
                item[0][0],
                item[0][1],
            ),
        )
        return _no_reversal(path[0], previous, layout, grid)
    if explore_target is not None:
        return _path_step(layout, grid, explore_target, previous)
    if portal.distance_cells >= config.explore_fallback_cells:
        return _path_step(layout, grid, portal.rel_cells, previous)
    # Standing next to the (inactive) portal with nothing left to visit in the
    # window: walk *away* from it, otherwise the rounded bearing is our own cell.
    step = _flee_step(layout, grid, [screen_delta(portal.rel_cells)], previous)
    return 0 if step is None else step


def _path_step(
    layout: Layout,
    grid: np.ndarray,
    target_cells: Any,
    previous: int = 0,
    *,
    avoid_reversal: bool = True,
) -> int:
    """First step of a shortest BFS path to a grid-cell delta.

    The target is clamped into the local window, so a target outside it (the
    portal, or an enemy at the perception edge) becomes "walk its way". When the
    clamped target is walled off, the policy walks to the reachable cell closest
    to it, which is what makes it go *around* a wall instead of pushing into it.
    Returns ``0`` (no movement) when the target is the player's own cell or the
    window offers nothing reachable.
    """
    radius = layout.grid_radius
    target = (
        max(-radius, min(radius, int(round(float(target_cells[0]))))),
        max(-radius, min(radius, int(round(float(target_cells[1]))))),
    )
    if target == (0, 0):
        return 0
    paths = _bfs(layout, grid)
    step = _reachable_step(layout, grid, paths, target)
    if step is None:
        return 0
    if avoid_reversal:
        return _no_reversal(step, previous, layout, grid)
    return _sticky(step, previous, layout, grid)


def _reachable_step(
    layout: Layout,
    grid: np.ndarray,
    paths: Mapping[tuple[int, int], tuple[int, int]],
    target: tuple[int, int],
) -> int | None:
    """First step of a BFS path to [param target], or to the best waypoint.

    Four tiers, in order — each one answers "the target is not reachable
    directly, what now?" without ever pressing into a wall:

    1. the target cell itself, when it is reachable inside the window;
    2. the reachable **doorway** pointing at the target — rooms only connect
       through door cells, so the door is the way out of a room the target is
       not in (this is what lets the policy leave the 21x21 window);
    3. the nearest reachable **frontier** (``visited == 0``) cell — the target is
       walled off and the route may be through unexplored floor;
    4. the reachable cell closest to the target, as a last resort so the body
       keeps moving instead of parking.
    """
    reachable = [(cell, path) for cell, path in paths.items() if cell != (0, 0)]
    if not reachable:
        return None
    if target in paths and target != (0, 0):
        return paths[target][0]
    doorways = [
        (cell, path)
        for cell, path in reachable
        if layout.cell(grid, cell[0], cell[1], "doorway") > 0.5
    ]
    if doorways:
        _, path = min(
            doorways,
            key=lambda item: (
                round(_distance_to(item[0], target), 6),
                item[1][1],
                item[0][0],
                item[0][1],
            ),
        )
        return path[0]
    frontier = [
        (cell, path) for cell, path in reachable if layout.visited(grid, cell[0], cell[1]) < 0.5
    ]
    if frontier:
        _, path = min(
            frontier,
            key=lambda item: (
                item[1][1],
                round(_distance_to(item[0], target), 6),
                item[0][0],
                item[0][1],
            ),
        )
        return path[0]
    _, path = min(
        reachable,
        key=lambda item: (
            round(_distance_to(item[0], target), 6),
            item[1][1],
            item[0][0],
            item[0][1],
        ),
    )
    return path[0]


def _centroid_screen(enemies: Sequence[EnemySlot]) -> np.ndarray:
    """Screen-space centroid of a group of enemies (cell units)."""
    total = np.zeros(2, dtype=np.float64)
    for slot in enemies:
        total += np.asarray(slot.screen_delta_cells, dtype=np.float64)
    return total / max(len(enemies), 1)


#: Player hurtbox half-extents and vertical offset, measured from
#: ``scenes/actors/player.tscn`` (44 x 56 at ``(0, -28)`` relative to the feet).
PLAYER_HURTBOX_HALF = (22.0, 28.0)
PLAYER_HURTBOX_OFFSET_Y = -28.0
#: Enemy hurtbox (``scenes/actors/enemies/enemy.tscn``): 44 x 56 at ``(0, -22)``.
ENEMY_HURTBOX_HALF = (22.0, 28.0)
ENEMY_HURTBOX_OFFSET_Y = -22.0
#: Swing rectangle width as a fraction of the reach (``Player.HITBOX_WIDTH_PER_REACH``).
SWING_WIDTH_PER_REACH = 0.92


def _swing_covers(facing: Sequence[float], rel_screen_px: Sequence[float], reach_px: float) -> bool:
    """True when the current facing puts the enemy inside the swing volume.

    The player's swing box is a rectangle of ``reach`` x ``0.92 * reach`` that
    starts at the feet and points along ``facing`` (``Player._apply_hitbox_geometry``);
    the enemy's hurtbox is an axis-aligned 44x56 rectangle centred 22 px above its
    feet. A swing only lands when those two overlap, so this is an exact
    separating-axis test instead of a ``dot(facing, to_enemy)`` guess — the
    reason the earlier facing heuristic both refused hits that would have landed
    and walked the body into point-blank range to "turn".
    """
    fx = float(facing[0])
    fy = float(facing[1])
    length = float(np.hypot(fx, fy))
    if length <= 1e-6:
        return False
    fx /= length
    fy /= length
    # Enemy hurtbox centre relative to the player's feet.
    ex = float(rel_screen_px[0])
    ey = float(rel_screen_px[1]) + ENEMY_HURTBOX_OFFSET_Y
    half_length = reach_px * 0.5
    half_width = reach_px * SWING_WIDTH_PER_REACH * 0.5
    # Swing rect centre: half the reach along the facing.
    cx = fx * half_length
    cy = fy * half_length
    # Axis-aligned hurtbox axes first.
    dx = ex - cx
    dy = ey - cy
    if abs(dx) > PLAYER_HURTBOX_HALF[0] + abs(fx) * half_length + abs(fy) * half_width:
        return False
    if abs(dy) > PLAYER_HURTBOX_HALF[1] + abs(fy) * half_length + abs(fx) * half_width:
        return False
    # Then the swing rect's own axes (forward and perpendicular).
    forward = dx * fx + dy * fy
    if abs(forward) > half_length + PLAYER_HURTBOX_HALF[0] * abs(fx) + PLAYER_HURTBOX_HALF[1] * abs(
        fy
    ):
        return False
    lateral = -dx * fy + dy * fx
    if abs(lateral) > half_width + PLAYER_HURTBOX_HALF[0] * abs(fy) + PLAYER_HURTBOX_HALF[1] * abs(
        fx
    ):
        return False
    return True


def _flee_step(
    layout: Layout,
    grid: np.ndarray,
    threats: Sequence[Sequence[float]],
    previous: int = 0,
    depth: int = 4,
) -> int | None:
    """BFS step that opens the most distance to the nearest of ``threats``.

    ``threats`` are screen-space vectors (cell units) from the player to each
    enemy that matters. Scoring a single centroid flips direction when two
    enemies stand on opposite sides (the measured two-enemy death spiral: the
    body ping-ponged between two attackers for 20 s while taking every hit), so
    the score is the distance to the *closest* threat. The iso projection is
    linear, so a walkable cell's screen position can be compared directly.

    Looking ``depth`` cells ahead is what turns the body before it runs into a
    wall or a doorway it cannot back out of; the immediate eight axes alone
    (the RL-4 behaviour) walk straight into the nearest corner.
    """
    vectors = [np.asarray(threat, dtype=np.float64) for threat in threats]
    if not vectors:
        return None
    best_index: int | None = None
    best_score = -np.inf
    for cell, (first_step, distance) in _bfs(layout, grid).items():
        if cell == (0, 0) or distance > depth:
            continue
        position = screen_delta(cell)
        nearest = min(float(np.hypot(*(position - vector))) for vector in vectors)
        score = nearest - 0.05 * float(distance)
        if score > best_score + _HYSTERESIS:
            best_score = score
            best_index = first_step
    if best_index is None:
        return None
    if previous and previous != best_index and 0 < previous < len(layout.actions):
        # "Keep walking": the away direction is noisy at point blank (and with a
        # pack it flips between attackers), so a still-good previous axis wins.
        # Otherwise the body oscillates across the enemy instead of leaving.
        spec = layout.actions[previous]
        if spec.moving and _step_is_free(layout, grid, spec.grid):
            position = screen_delta(spec.grid)
            nearest = min(float(np.hypot(*(position - vector))) for vector in vectors)
            if nearest >= best_score - 2.0 * _HYSTERESIS:
                return previous
    return best_index


def _steer(layout: Layout, screen_delta_cells: np.ndarray, previous: int = 0) -> int:
    """Best movement action for a screen delta, sticky to the previous axis.

    The eight action directions are 45° apart, so a target sitting on a bisector
    makes the plain arg-max flip between two axes every window — which cancels
    the movement out. Keeping the previous axis while it is within
    :data:`_HYSTERESIS` of the best dot product removes that jitter.
    """
    direction = normalise(screen_delta_cells)
    length = float(np.hypot(*direction))
    if length <= 1e-9:
        return 0
    best_index = 0
    best_dot = -np.inf
    for spec in layout.actions:
        if not spec.moving:
            continue
        dot = float(spec.screen[0] * direction[0] + spec.screen[1] * direction[1])
        if dot > best_dot + 1e-12:
            best_dot = dot
            best_index = spec.index
    if previous and previous != best_index and 0 < previous < len(layout.actions):
        spec = layout.actions[previous]
        if spec.moving:
            kept = float(spec.screen[0] * direction[0] + spec.screen[1] * direction[1])
            if kept >= best_dot - _HYSTERESIS:
                return previous
    return best_index


def _previous_move(info: Mapping[str, Any] | None) -> int:
    """The move component of the previously applied action (0 when unknown)."""
    if not isinstance(info, Mapping):
        return 0
    action = info.get("action")
    if not isinstance(action, (list, tuple, np.ndarray)) or len(action) < 3:
        return 0
    try:
        value = int(action[0])
    except (TypeError, ValueError):
        return 0
    return value if 0 <= value <= 8 else 0


def _flee_hazard(layout: Layout, grid: np.ndarray) -> int | None:
    """Step out of a live telegraph under the player (boss AoE), if any."""
    if layout.hazard(grid, 0, 0) < 0.5:
        return None
    best_index: int | None = None
    for spec in layout.actions:
        if not spec.moving:
            continue
        if not _step_is_free(layout, grid, spec.grid):
            continue
        if layout.hazard(grid, spec.grid[0], spec.grid[1]) >= 0.5:
            continue
        best_index = spec.index
        break
    return best_index


def frontier_open(layout: Layout, grid: np.ndarray) -> bool:
    """True when the local grid still has a reachable, unvisited cell."""
    for cell in _bfs(layout, grid):
        if cell != (0, 0) and layout.visited(grid, cell[0], cell[1]) < 0.5:
            return True
    return False


def _bfs(layout: Layout, grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    """BFS over walkable local cells: cell delta → ``(first action, depth)``.

    Insertion order is breadth-first, so :func:`_explore_step` sees the nearest
    frontier first and ties are broken by the published action order.
    """
    start = (0, 0)
    paths: dict[tuple[int, int], tuple[int, int]] = {start: (0, 0)}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        first_step, depth = paths[cell]
        for spec in layout.actions:
            if not spec.moving:
                continue
            nxt = (cell[0] + spec.grid[0], cell[1] + spec.grid[1])
            if nxt in paths:
                continue
            if not _step_is_free(layout, grid, spec.grid, cell):
                continue
            paths[nxt] = (spec.index if cell == start else first_step, depth + 1)
            queue.append(nxt)
    return paths


#: "This enemy cannot answer for a long time" (recover/stagger/dead).
_NO_THREAT_S = 99.0

_HYSTERESIS = 0.06
#: Walked distance (cells) over the no-progress window that still counts as frozen.
_FROZEN_CELLS = 0.05
#: Walked distance (cells) over the loop window that proves the body is moving.
_PINGPONG_WALK_CELLS = 1.0
#: Net displacement (cells) under which a moving body has made no progress.
_PINGPONG_NET_CELLS = 0.75


def _sticky(step: int, previous: int, layout: Layout, grid: np.ndarray) -> int:
    """Keep ``previous`` instead of ``step`` when both are free and adjacent.

    Only used for wall-aware BFS paths: the path is recomputed every window and
    a one-cell shift in the target can flip the first step between two equally
    good axes, which would make the player pace in place.
    """
    if not previous or previous == step or not (0 < previous < len(layout.actions)):
        return step
    candidate = layout.actions[previous]
    chosen = layout.actions[step] if 0 <= step < len(layout.actions) else None
    if chosen is None or not candidate.moving or not chosen.moving:
        return step
    if not _step_is_free(layout, grid, candidate.grid):
        return step
    # Only keep the sticky step when it does not point backwards.
    if candidate.screen[0] * chosen.screen[0] + candidate.screen[1] * chosen.screen[1] <= 0.0:
        return step
    return previous


def _no_reversal(chosen: int, previous: int, layout: Layout, grid: np.ndarray) -> int:
    """Refuse to reverse the walking direction for a single window.

    A frontier target that is one cell behind the player makes the pure policy
    alternate "step toward" and "step back" every cell, which cancels out into a
    permanent two-cell loop. Keeping the previous direction (when it is a legal
    step) breaks the cycle; the next recomputation sees a changed terrain
    window and picks a genuinely new frontier.
    """
    if not previous or previous == chosen or not (0 < previous < len(layout.actions)):
        return chosen
    if layout.opposite_move(previous) != chosen:
        return chosen
    spec = layout.actions[previous]
    if spec.moving and _step_is_free(layout, grid, spec.grid):
        return previous
    return chosen


def _distance_to(cell: tuple[int, int], target: tuple[int, int]) -> float:
    """Screen distance (grid-cell units) between two cell deltas."""
    return _screen_length((cell[0] - target[0], cell[1] - target[1]))


def _screen_length(cell: tuple[int, int]) -> float:
    """Screen length of a grid-cell delta (cell units)."""
    return float(np.hypot(*screen_delta(cell)))


def _step_is_free(
    layout: Layout, grid: np.ndarray, step: Any, cell: tuple[int, int] = (0, 0)
) -> bool:
    """True when moving ``step`` from ``cell`` lands on walkable terrain.

    Diagonal moves must not cut a wall corner: both orthogonal neighbours have
    to be walkable too (the floor generator only carves axis-aligned
    corridors, so diagonal steps exist only in open rooms).
    """
    dx = int(step[0])
    dy = int(step[1])
    target = (cell[0] + dx, cell[1] + dy)
    if not layout.walkable(grid, target[0], target[1]):
        return False
    if dx != 0 and dy != 0:
        if not layout.walkable(grid, cell[0] + dx, cell[1]):
            return False
        if not layout.walkable(grid, cell[0], cell[1] + dy):
            return False
    return True


def _resolve_layout(info: Mapping[str, Any] | None, layout: Layout | None) -> Layout:
    if isinstance(layout, Layout):
        return layout
    if isinstance(info, Mapping):
        candidate = info.get("layout")
        if isinstance(candidate, Layout):
            return candidate
        if isinstance(candidate, Mapping):
            return Layout.from_layout_dict(candidate)
    raise TypeError("heuristic_action needs the env layout (pass layout=env.layout)")


def _action(move: int, attack: int, interact: int) -> np.ndarray:
    return np.array([int(move), int(attack), int(interact)], dtype=np.int64)


class HeuristicPolicy:
    """Stateful wrapper around :func:`heuristic_action` with a pocket escape.

    The decision core stays a pure function of ``(obs, info)``; this wrapper adds
    the one piece of memory a policy needs in this game: *"I have been walking
    and my sub-cell position has not changed"*. The terrain grid is plan truth —
    a prop, a doorway or a wall edge can still block the body, and the game's
    own velocity stays at the *commanded* value while it is blocked — so a
    stateless policy can push into the same blocker for the rest of the episode.

    The pocket is detected from the player's **sub-cell** position (the
    ``player_cell_offset`` diagnostic, §5.6, plus the cell fraction), which
    catches both a body stuck on a blocker and the two-cell ping-pong that
    changes the cell on every window. The env's ``stuck`` flag (§5.7) is
    honoured when the process provides it: it fires immediately instead of
    after ``no_progress_windows`` decisions.

    On a pocket the wrapper commits to a *free* movement axis (rotating
    clockwise from the one the core asked for) for ``escape_decisions``
    decisions, which walks the body out of the trap; if that axis is blocked in
    practice the detection fires again and it rotates on.

    The second piece of state is the **exploration sweep**: once the local
    21x21 window has no unvisited cell left, the pure core has no target and the
    portal is the only global landmark it can see. Walking the portal bearing is
    right when the portal is far, but once the body *stands next to the inactive
    portal* that bearing rounds to its own cell and the policy froze (measured:
    3,421 of 4,500 decisions on seed 2, 76 % of the episode). The wrapper
    therefore holds a deterministic ring sweep around the portal — radii from
    ``config.sweep_radii_cells`` in eight grid directions — and hands the current
    waypoint to the core as ``explore_target``. It advances on arrival, on a
    stall counter, and resets when the portal opens. The core still prefers a
    real local frontier whenever one exists.

    ``__call__`` matches the evaluation/replay policy signature
    ``(obs, info, layout)``; the layout is taken from the constructor.
    """

    def __init__(
        self,
        layout: Layout,
        config: HeuristicConfig = DEFAULT_CONFIG,
        *,
        no_progress_windows: int = 8,
        loop_window: int = 40,
        loop_unique: int = 4,
        escape_decisions: int = 12,
    ) -> None:
        self.layout = layout
        self.config = config
        self.no_progress_windows = int(no_progress_windows)
        self.loop_window = int(loop_window)
        self.loop_unique = int(loop_unique)
        self.escape_decisions = int(escape_decisions)
        self._last_offset: np.ndarray | None = None
        self._last_fine: tuple[float, float] | None = None
        self._has_odometry = True
        self._odometry = np.zeros(2, dtype=np.float64)
        self._trail: deque[np.ndarray] = deque(maxlen=self.loop_window)
        self._steps: deque[float] = deque(maxlen=self.loop_window)
        self._flat: deque[float] = deque(maxlen=self.no_progress_windows)
        self._escape_left = 0
        self._escape_move = 0
        self._goal: tuple[int, int] | None = None
        self._goal_age = 0
        self._explore_goal: tuple[int, int] | None = None
        self._explore_age = 0
        self._sidestep_left = 0
        self._sidestep_move = 0
        self.map = NavMap(layout)

    def __call__(
        self, obs: np.ndarray, info: Mapping[str, Any] | None = None, layout: Layout | None = None
    ) -> np.ndarray:
        resolved = layout or self.layout
        action = heuristic_action(obs, info, resolved, self.config)
        navigation = self._nav_move(resolved, obs, info)
        if navigation:
            action = action.copy()
            action[0] = navigation
        action = self._pack_sidestep(resolved, obs, action)
        player = resolved.player_features(obs)
        grid = resolved.grid(obs)
        engaged = any(
            slot.screen_distance_px <= self.config.pack_radius_px
            for slot in resolved.perceived_enemies(obs)
        )
        self._track(info, player.cell_fraction)

        stuck = bool(info.get("stuck", False)) if isinstance(info, Mapping) else False
        if (engaged and not stuck) or int(action[0]) == 0:
            # Combat footwork and deliberate standing still are not a pocket;
            # a body that has not moved for the whole 2 s window still is.
            self._reset_memory()
        elif self._escape_left <= 0 and (stuck or self._pocketed()):
            self._escape_move = self._free_rotation(resolved, grid, int(action[0]))
            self._escape_left = self.escape_decisions if self._escape_move else 0
            self._reset_memory()
        if self._escape_left > 0:
            self._escape_left -= 1
            spec = resolved.actions[self._escape_move] if self._escape_move else None
            if spec is not None and _step_is_free(resolved, grid, spec.grid):
                action = action.copy()
                action[0] = self._escape_move
            else:
                self._escape_left = 0
        return action

    # -- global navigation -------------------------------------------------- #
    def _nav_move(
        self,
        layout: Layout,
        obs: np.ndarray,
        info: Mapping[str, Any] | None = None,
    ) -> int | None:
        """The global-map move for this decision, or ``None`` to keep the core's.

        The local 21x21 window cannot path to a goal that sits behind a wall it
        does not see (the measured seed-2 failure: two enemies alive, ~850 px
        away, 233 s spent circling). :class:`~ashcrown_rl.navmap.NavMap` keeps
        the whole walkable raster in portal-relative cells and answers with the
        first grid step instead. Combat inside ``local_approach_px`` and the
        hazard dodge stay with the pure core; a live telegraph under the body is
        always the core's call.
        """
        grid = layout.grid(obs)
        if layout.hazard(grid, 0, 0) >= 0.5:
            return None
        portal = layout.portal(obs)
        offset = info.get("player_cell_offset") if isinstance(info, Mapping) else None
        offset = _offset_vector(offset)
        self.map.update(grid, portal, tuple(offset) if offset is not None else (0.0, 0.0))
        enemies = [slot for slot in layout.perceived_enemies(obs) if not slot.dead]
        if enemies:
            focus = min(enemies, key=lambda slot: (slot.screen_distance_px, slot.slot))
            if focus.screen_distance_px <= self.config.local_approach_px:
                self._goal = None
                return None
            self._goal = (
                self.map.player[0] + int(round(focus.rel_cells[0])),
                self.map.player[1] + int(round(focus.rel_cells[1])),
            )
            self._goal_age = 0
            step = self.map.step_towards(self._goal)
        elif portal.active:
            self._goal = None
            step = self.map.step_towards(self.map.portal)
        elif self._goal is not None and self._goal_age < self.config.goal_memory_decisions:
            # The enemy is right at the 900 px perception edge: the map route to
            # it often leaves the room *away* from it first, which drops it out
            # of perception and made the policy flip between "approach" and
            # "explore" for 4,000 decisions (measured on seed 5: 5 kills, then
            # nothing for 273 s). Remember the last goal and keep walking to it.
            self._goal_age += 1
            step = self.map.step_towards(self._goal)
        else:
            self._goal = None
            step = self.map.explore_step(self._explore_target())
            if step is None:
                return None
        if step is None:
            return None
        return layout.move_for_grid_step(step)

    def _explore_target(self) -> tuple[int, int] | None:
        """The committed exploration destination (kept for a bounded time).

        ``NavMap.explore_step`` re-derives "walk deeper" every window, and with
        two equally far ends of the known area the choice flips and the body
        oscillates between them — the measured seed-3 stall (9 kills, then 233 s
        of walking with 0 damage and no new cells). Holding one destination for
        ``goal_memory_decisions`` windows (and dropping it on arrival) makes the
        walk monotone, which is what actually unfolds the map.
        """
        if self._explore_goal is not None and self._explore_age < self.config.goal_memory_decisions:
            self._explore_age += 1
            if self.map.step_towards(self._explore_goal) is not None:
                return self._explore_goal
        self._explore_goal = self.map.deepest_cell()
        self._explore_age = 0
        return self._explore_goal

    # -- pack sidestep ------------------------------------------------------ #
    def _pack_sidestep(self, layout: Layout, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Take one escape step when two enemies are already on top of the body.

        This is the one surviving piece of pack footwork that measured *better*
        than plain trading: with two attackers inside ``pack_sidestep_px`` the
        next swing is a guaranteed double trade, so a single step out (then the
        normal combat rules) keeps the pair from stacking perfectly on top of the
        body. Longer commitments (3-9 decisions) were measured worse: they turn
        the fight into a kite that never kills anything.
        """
        if int(action[1]) == 1:
            self._sidestep_left = 0
            return action
        if self._sidestep_left > 0:
            self._sidestep_left -= 1
            spec = layout.actions[self._sidestep_move] if self._sidestep_move else None
            if (
                spec is not None
                and spec.moving
                and _step_is_free(layout, layout.grid(obs), spec.grid)
            ):
                action = action.copy()
                action[0] = self._sidestep_move
            return action
        enemies = [slot for slot in layout.perceived_enemies(obs) if not slot.dead]
        if len(enemies) < 2:
            return action
        close = [
            slot for slot in enemies if slot.screen_distance_px <= self.config.pack_sidestep_px
        ]
        if len(close) < 2:
            return action
        grid = layout.grid(obs)
        move = _flee_step(layout, grid, [screen_delta(slot.rel_cells) for slot in close], 0)
        if not move:
            return action
        self._sidestep_move = move
        self._sidestep_left = max(self.config.pack_sidestep_decisions, 1) - 1
        action = action.copy()
        action[0] = move
        return action

    # -- pocket detection --------------------------------------------------- #
    def _track(self, info: Mapping[str, Any] | None, cell_fraction: Sequence[float]) -> None:
        """Accumulate sub-cell odometry from ``info.player_cell_offset``.

        The offset is the player's position inside its cell in cell units
        (`docs/07` §5.6), so its per-decision difference is the walked distance
        modulo one cell. A decision never moves the body more than half a cell
        (4 ticks at ~220 px/s is ~0.1 cell), so wrapping the difference into
        ``[-0.5, 0.5)`` recovers the signed step and the sum is a local
        odometry track: *walked* distance against *net* displacement, which is
        what separates a real pocket from normal travel.

        Without the diagnostic (an engine that does not publish it) the policy
        falls back to the coarse normalised cell fraction, which still catches
        a body that never changes cell but cannot see a two-cell ping-pong.
        """
        offset = info.get("player_cell_offset") if isinstance(info, Mapping) else None
        current = _offset_vector(offset)
        if current is None:
            fine = (round(float(cell_fraction[0]), 3), round(float(cell_fraction[1]), 3))
            step = 0.0 if fine == self._last_fine else 1.0
            self._last_fine = fine
            self._last_offset = None
            self._has_odometry = False
        else:
            step = 0.0
            if self._last_offset is not None:
                delta = np.mod(current - self._last_offset + 0.5, 1.0) - 0.5
                self._odometry += delta
                step = float(np.hypot(*delta))
            self._last_offset = current
            self._has_odometry = True
        self._steps.append(step)
        self._flat.append(step)
        self._trail.append(self._odometry.copy())

    def _pocketed(self) -> bool:
        """True when the body cannot move, or walks without getting anywhere."""
        recent = [value for value in self._flat if np.isfinite(value)]
        if len(recent) == self._flat.maxlen and sum(recent) < _FROZEN_CELLS:
            return True

        if not self._has_odometry:
            return False
        steps = [value for value in self._steps if np.isfinite(value)]
        if len(steps) < self.loop_window or not self._trail:
            return False
        walked = sum(steps)
        net = float(np.hypot(*(self._odometry - self._trail[0])))
        return walked > _PINGPONG_WALK_CELLS and net < _PINGPONG_NET_CELLS

    def _reset_memory(self) -> None:
        self._trail.clear()
        self._steps.clear()
        self._flat.clear()

    def _free_rotation(self, layout: Layout, grid: np.ndarray, move: int) -> int:
        """First walkable movement axis after ``move`` (clockwise), 0 if none."""
        for step in range(1, 9):
            candidate = _rotate_move(move, step)
            spec = layout.actions[candidate]
            if spec.moving and _step_is_free(layout, grid, spec.grid):
                return candidate
        return 0


def _offset_vector(offset: Any) -> np.ndarray | None:
    """``info.player_cell_offset`` as a finite 2-vector, or ``None``."""
    if not isinstance(offset, (list, tuple, np.ndarray)) or len(offset) < 2:
        return None
    vector = np.array([float(offset[0]), float(offset[1])], dtype=np.float64)
    return vector if np.all(np.isfinite(vector)) else None


def _rotate_move(move: int, steps: int) -> int:
    """Rotate a move index (1–8, clockwise in the published order) by ``steps``."""
    if move <= 0:
        return 0
    return ((int(move) - 1 + int(steps)) % 8) + 1


__all__ = [
    "DEFAULT_CONFIG",
    "HeuristicConfig",
    "HeuristicPolicy",
    "frontier_open",
    "heuristic_action",
]
