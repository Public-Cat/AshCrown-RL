"""Scripted heuristic baseline for ``ashcrown-obs-v1`` (RL-4, ``docs/07`` §9.4).

The policy is a **pure function of the observation** (:func:`heuristic_action`),
so it can be unit-tested without a Godot process and replayed exactly:

1. **Fight.** Hunt the nearest perceived enemy: walk to it with a BFS path over
   the local terrain grid, swing from just outside its reach, back off while it
   telegraphs and strike during its recovery. With two or more enemies adjacent
   (or at low HP with company) it disengages first instead of trading blows.
2. **Portal.** Once the floor is cleared, path to the portal and pulse
   ``interact`` inside its cell.
3. **Explore.** While the floor is not cleared and nothing is perceived, walk
   the local-grid frontier: BFS to the nearest ``visited == 0`` floor cell
   (tie-broken toward the portal side). When the whole 21x21 window is explored
   the target falls outside the window, so the policy follows the portal
   bearing instead — the portal sits on the far side of the floor, so that
   route sweeps the map's long axis.

Every tie-break is a fixed action-index/dict order and no RNG is used, so the
same observation always yields the same action. Acceptance (``docs/07`` §9.4):
clears level 1 on at least 8 of 10 fixed seeds with the warrior inside the
5-game-minute limit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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

    #: Extra pixels on the player's weapon reach before it swings (hurtbox).
    attack_slack_px: float = 26.0
    #: Hard cap on the swing start distance: the swing is committed, so a start
    #: beyond the box's real coverage only locks the player in place.
    max_attack_px: float = 69.0
    #: Extra pixels on the enemy's reach that count as "dangerously close".
    retreat_slack_px: float = 34.0
    #: Minimum ``dot(facing, to_enemy)`` before swinging (the swing locks facing).
    facing_cos: float = 0.8
    #: Swing only from at least this distance (``0`` = swing at any range).
    min_attack_px: float = 60.0
    #: Post-swing cooldown (seconds) that makes a close-range swing safe.
    safe_cooldown_s: float = 0.3
    #: Enemies inside this radius are "adjacent" (they can pile on together).
    group_radius_px: float = 300.0
    #: Wounded enough that a mob cannot be out-traded → disengage instead.
    disengage_hp_frac: float = 0.35
    #: Portal distance (iso cells) at which ``interact`` is pulsed.
    portal_interact_cells: float = 0.7


DEFAULT_CONFIG = HeuristicConfig()


def heuristic_action(
    obs: np.ndarray,
    info: Mapping[str, Any] | None = None,
    layout: Layout | None = None,
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Return ``[move, attack, interact]`` for one observation.

    ``layout`` is the parsed handshake layout (``env.layout``). When omitted it
    is taken from ``info["layout"]`` (a :class:`Layout`, or a raw handshake
    ``layout`` mapping), so the function also works in a process-free test.
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
    return _action(_explore_step(resolved, grid, portal, previous), 0, 0)


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

    ``enemies`` is nearest-first. The duel is a three-beat loop: close to just
    inside the player's reach, swing while the enemy cannot answer (recovering,
    staggered or still on its post-swing cooldown), and step out while it is
    winding up. The **nearest wounded** enemy is the focus target — spreading
    damage over a pack leaves three enemies hitting as hard as three healthy
    ones — and a mob is only dodged as a group while the player is too hurt to
    trade.
    """
    close = [slot for slot in enemies if slot.screen_distance_px <= config.group_radius_px]
    focus = close or [enemies[0]]
    enemy = min(focus, key=lambda slot: (round(slot.hp_frac, 3), slot.screen_distance_px))
    outnumbered = len(close) >= 2
    if outnumbered and player.hp_frac <= config.disengage_hp_frac:
        # Wounded and surrounded: back out of the group and let it string out
        # instead of feeding the whole pack one hit at a time.
        escape = _retreat_step(layout, grid, _centroid_screen(close), previous)
        if escape is not None:
            return _action(escape, 0, 0)

    if len(close) < 2:
        # One enemy: the duel dance (dodge the telegraph, strike the recovery).
        threat = _incoming_threat(enemies, config)
        if threat is not None:
            escape = _retreat_step(layout, grid, screen_delta(threat.rel_cells), previous)
            if escape is not None:
                return _action(escape, 0, 0)
            # Cornered: swing back rather than stand in the telegraph.
            return _action(0, 1, 0)

    screen = screen_delta(enemy.rel_cells)
    distance_px = float(np.hypot(*screen)) * PX_PER_CELL
    toward = _steer(layout, screen, previous)
    in_reach = distance_px <= min(
        player.attack_reach_px + config.attack_slack_px, config.max_attack_px
    )

    # A queued strike chains at the end of the current swing, which keeps the
    # player locked in the attack state (movement is ignored there). Never
    # queue: one pulse per swing keeps a free window for footwork.
    if player.state == "attack" or player.attack_queued:
        return _action(0, 0, 0)

    threatened = enemy.state in ("windup", "attack") or enemy.hitbox_active
    if threatened and distance_px <= enemy.attack_reach_px + config.retreat_slack_px:
        if outnumbered and in_reach and _facing_aligned(player, screen, config.facing_cos):
            # A mob leaves no gap to dodge through: retreating only bleeds HP
            # without dealing damage, so trade inside the pack and finish one
            # enemy at a time instead.
            return _action(0, 1, 0)
        escape = _retreat_step(layout, grid, screen, previous)
        if escape is not None:
            return _action(escape, 0, 0)
        # Cornered: trade the hit rather than stand still (facing is current
        # because the enemy is right in front of us).
        return _action(toward, 1, 0)

    if in_reach:
        # The enemy reaches further than the player and its wind-up is shorter
        # than a swing, so only start one from (a) outside its attack trigger
        # range or (b) while it cannot answer: recovering, staggered or still
        # on its post-swing cooldown. Swinging from point blank otherwise is a
        # guaranteed trade, and 12 trades is the whole health bar.
        safe = (
            outnumbered
            or enemy.state in ("recover", "stagger")
            or enemy.attack_cooldown_s >= config.safe_cooldown_s
        )
        if distance_px >= config.min_attack_px or safe:
            if _facing_aligned(player, screen, config.facing_cos):
                return _action(0, 1, 0)
        else:
            escape = _retreat_step(layout, grid, screen, previous)
            if escape is not None:
                return _action(escape, 0, 0)

    move = _path_step(layout, grid, enemy.rel_cells, previous)
    if move == 0 and distance_px > 1.0:
        # Inside half a grid cell the rounded BFS target is our own cell; steer
        # with the exact screen direction instead (this is what turns the
        # facing, and the swing locks the facing).
        move = toward
    return _action(move, 0, 0)


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
    layout: Layout, grid: np.ndarray, portal: PortalFeatures, previous: int = 0
) -> int:
    """BFS first step to the nearest unexplored floor cell of the local window.

    The env marks a 3x3 brush around the player as ``visited``, so the nearest
    unvisited cell is always just outside the brush and the walk sweeps the
    floor systematically. Ties (the eight cells of the same BFS ring) go to the
    cell closest to the portal: the exit sits on the far side of the floor, so
    the sweep leans across the map instead of orbiting the room it started in.

    When the whole window is explored there is no frontier left and the target
    is outside the 21x21 window, so the portal bearing takes over — that is the
    long-range fallback the spec asks for (§5.7).
    """
    paths = _bfs(layout, grid)
    frontier = [
        (cell, path)
        for cell, path in paths.items()
        if cell != (0, 0) and layout.visited(grid, cell[0], cell[1]) < 0.5
    ]
    if not frontier:
        return _path_step(layout, grid, portal.rel_cells, previous)
    portal_cell = (
        int(round(float(portal.rel_cells[0]))),
        int(round(float(portal.rel_cells[1]))),
    )
    _, path = min(
        frontier,
        key=lambda item: (
            item[1][1],
            round(_distance_to(item[0], portal_cell), 6),
            item[0][0],
            item[0][1],
        ),
    )
    return _no_reversal(path[0], previous, layout, grid)


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


def _incoming_threat(enemies: Sequence[EnemySlot], config: HeuristicConfig) -> EnemySlot | None:
    """The nearest enemy whose swing is already telegraphing inside its reach."""
    for enemy in enemies:
        if not (enemy.state in ("windup", "attack") or enemy.hitbox_active):
            continue
        if enemy.screen_distance_px <= enemy.attack_reach_px + config.retreat_slack_px:
            return enemy
    return None


def _facing_aligned(player: PlayerFeatures, screen: np.ndarray, threshold: float) -> bool:
    """True when the player already faces the enemy (the swing locks facing)."""
    facing = np.asarray(player.facing, dtype=np.float64)
    if float(np.hypot(*facing)) <= 1e-6:
        return False
    to_enemy = normalise(screen)
    if float(np.hypot(*to_enemy)) <= 1e-6:
        return True
    return float(facing[0] * to_enemy[0] + facing[1] * to_enemy[1]) >= threshold


def _retreat_step(
    layout: Layout,
    grid: np.ndarray,
    threat_screen: np.ndarray,
    previous: int = 0,
) -> int | None:
    """Walkable step away from a threat, or ``None`` when there is no escape."""
    away = normalise(-np.asarray(threat_screen, dtype=np.float64))
    best_index: int | None = None
    best_score = -np.inf
    for spec in layout.actions:
        if not spec.moving:
            continue
        if not _step_is_free(layout, grid, spec.grid):
            continue
        score = float(spec.screen[0] * away[0] + spec.screen[1] * away[1])
        if score > best_score + _HYSTERESIS:
            best_score = score
            best_index = spec.index
    if best_index is not None and previous and previous != best_index:
        # Keep the previous escape direction when it is almost as good: flipping
        # between two adjacent axes every window moves the player nowhere.
        spec = layout.actions[previous] if previous < len(layout.actions) else None
        if spec is not None and spec.moving and _step_is_free(layout, grid, spec.grid):
            kept = float(spec.screen[0] * away[0] + spec.screen[1] * away[1])
            if kept >= best_score - _HYSTERESIS:
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

    def __call__(
        self, obs: np.ndarray, info: Mapping[str, Any] | None = None, layout: Layout | None = None
    ) -> np.ndarray:
        resolved = layout or self.layout
        action = heuristic_action(obs, info, resolved, self.config)
        player = resolved.player_features(obs)
        grid = resolved.grid(obs)
        engaged = any(
            slot.screen_distance_px <= self.config.group_radius_px
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
