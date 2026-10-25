"""Scripted heuristic baseline for ``ashcrown-obs-v1`` (RL-4, ``docs/07`` §9.4).

The policy is a **pure function of the observation** (:func:`heuristic_action`),
so it can be unit-tested without a Godot process and replayed exactly:

1. **Fight.** Face/move toward the nearest perceived enemy; attack when the
   enemy is inside the player's reach; back off while the enemy is winding up
   (its reach is longer than the player's, so trading is the only way to lose
   HP) and step back in during the recovery.
2. **Portal.** Once the floor is cleared, follow the portal's grid delta with a
   wall-aware BFS over the local terrain grid instead of walking into walls.
3. **Explore.** With no perceived enemy and no portal, walk to the nearest
   unvisited walkable cell (the env-maintained ``visited`` channel is the
   policy's memory), which sweeps the floor until the next enemy is inside the
   900 px perception radius.

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

from .spaces import PX_PER_CELL, EnemySlot, Layout, PlayerFeatures, normalise, screen_delta


@dataclass(frozen=True)
class HeuristicConfig:
    """Full configuration of the scripted baseline (immutable, hashable)."""

    #: Extra pixels on the player's weapon reach before it swings (enemy hurtbox).
    attack_slack_px: float = 24.0
    #: Extra pixels on the enemy's reach that count as "dangerously close".
    retreat_slack_px: float = 16.0
    #: Minimum ``dot(facing, to_enemy)`` before swinging (the swing locks facing).
    facing_cos: float = 0.8
    #: Swing only from at least this distance (``0`` = swing at any range).
    min_attack_px: float = 56.0
    #: Post-swing cooldown (seconds) that makes a close-range swing safe.
    safe_cooldown_s: float = 0.3
    #: Portal distance (iso cells) at which ``interact`` is pulsed.
    portal_interact_cells: float = 0.7
    #: Portal distance (iso cells) at which the policy stops walking in.
    portal_brake_cells: float = 0.35


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
        move = _step_toward(resolved, grid, portal.rel_cells, previous)
        interact = 1 if portal.distance_cells <= config.portal_interact_cells else 0
        brake = portal.distance_cells <= config.portal_brake_cells
        return _action(0 if brake else move, 0, interact)

    return _action(_explore_step(resolved, grid, previous), 0, 0)


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

    ``enemies`` is nearest-first. Any enemy that is telegraphing a swing inside
    its own reach is a threat — including one that is not the nearest — so the
    policy dodges the most dangerous one instead of tunnel-visioning on the
    closest target.
    """
    threat = _incoming_threat(enemies, config)
    if threat is not None:
        escape = _retreat_step(layout, grid, screen_delta(threat.rel_cells), previous)
        return _action(0 if escape is None else escape, 0, 0)
    enemy = enemies[0]
    screen = screen_delta(enemy.rel_cells)
    distance_px = float(np.hypot(*screen)) * PX_PER_CELL
    toward = _steer(layout, screen, previous)
    in_reach = distance_px <= player.attack_reach_px + config.attack_slack_px

    # A queued strike chains at the end of the current swing, which keeps the
    # player locked in the attack state (movement is ignored there). Never
    # queue: one pulse per swing keeps a free window for footwork.
    if player.state == "attack" or player.attack_queued:
        return _action(0, 0, 0)

    threatened = enemy.state in ("windup", "attack") or enemy.hitbox_active
    if threatened and distance_px <= enemy.attack_reach_px + config.retreat_slack_px:
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
            enemy.state in ("recover", "stagger")
            or enemy.attack_cooldown_s >= config.safe_cooldown_s
        )
        if distance_px >= config.min_attack_px or safe:
            if _facing_aligned(player, screen, config.facing_cos):
                return _action(0, 1, 0)
        else:
            escape = _retreat_step(layout, grid, screen, previous)
            if escape is not None:
                return _action(escape, 0, 0)

    move = _step_toward(layout, grid, enemy.rel_cells, previous)
    if move == 0 and distance_px > 1.0:
        # Inside half a grid cell the rounded BFS target is our own cell; steer
        # with the exact screen direction instead (this is what turns the
        # facing, and the swing locks the facing).
        move = toward
    return _action(move, 0, 0)


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


def _step_toward(layout: Layout, grid: np.ndarray, target_cells: Any, previous: int = 0) -> int:
    """Wall-aware first step toward a grid-cell delta (BFS over the local grid).

    When the target is inside the window and reachable, the step is on a
    shortest path. When it is outside the window or walled off, the policy
    walks to the reachable **frontier** (unvisited) cell closest to the target:
    that is what makes it wall-aware — it walks *around* a wall instead of
    pushing into it, and the env-maintained ``visited`` channel keeps the sweep
    moving until the direct path opens.
    """
    target = (int(round(float(target_cells[0]))), int(round(float(target_cells[1]))))
    radius = layout.grid_radius
    target = (max(-radius, min(radius, target[0])), max(-radius, min(radius, target[1])))
    if target == (0, 0):
        return 0
    paths = _bfs(layout, grid)
    if target in paths:
        return _sticky(paths[target][0], previous, layout, grid)
    frontier = [
        (cell, path)
        for cell, path in paths.items()
        if cell != (0, 0) and layout.visited(grid, cell[0], cell[1]) < 0.5
    ]
    if frontier:
        _, path = min(
            frontier,
            key=lambda item: (
                round(_distance_to(item[0], target), 6),
                item[1][1],
                item[0][0],
                item[0][1],
            ),
        )
        return path[0]
    # Fully explored / boxed in: head to the reachable cell closest to the
    # target side of the window instead of standing still.
    candidates = [(cell, path) for cell, path in paths.items() if cell != (0, 0)]
    if not candidates:
        return 0
    _, path = min(
        candidates,
        key=lambda item: (
            round(_distance_to(item[0], target), 6),
            round(_screen_length(item[0]), 6),
            item[0][0],
            item[0][1],
        ),
    )
    return path[0]


def _explore_step(layout: Layout, grid: np.ndarray, previous: int = 0) -> int:
    """First step toward the nearest unvisited walkable cell (frontier search)."""
    paths = _bfs(layout, grid)
    for cell, path in paths.items():
        if cell == (0, 0):
            continue
        if layout.visited(grid, cell[0], cell[1]) < 0.5:
            return path[0]
    # Fully explored window (or boxed in): push to the farthest reachable cell.
    candidates = [(cell, path) for cell, path in paths.items() if cell != (0, 0)]
    if not candidates:
        return 0
    _, path = max(
        candidates,
        key=lambda item: (
            round(_screen_length(item[0]), 6),
            -item[0][0],
            -item[0][1],
        ),
    )
    return path[0]


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
    """Stateful wrapper around :func:`heuristic_action` with an anti-stuck guard.

    The decision core stays a pure function of ``(obs, info)``; this wrapper
    only adds the one piece of memory a policy needs in this game: *"I have
    been walking for a while and my cell has not changed"*. The world is not
    fully reflected in the observation — a prop or a doorstep can block the
    cell the terrain grid calls walkable — and with no memory the pure policy
    would push into that blocker forever. When ``no_progress`` consecutive
    windows produce no cell change while exploring, the wrapper rotates the
    requested direction by one action step and keeps rotating until the player
    moves again.

    ``__call__`` matches the evaluation/replay policy signature
    ``(obs, info, layout)``; the layout is taken from the constructor.
    """

    def __init__(
        self,
        layout: Layout,
        config: HeuristicConfig = DEFAULT_CONFIG,
        *,
        no_progress_windows: int = 12,
    ) -> None:
        self.layout = layout
        self.config = config
        self.no_progress_windows = int(no_progress_windows)
        self._last_cell: tuple[float, float] | None = None
        self._waiting = 0
        self._rotation = 0

    def __call__(
        self, obs: np.ndarray, info: Mapping[str, Any] | None = None, layout: Layout | None = None
    ) -> np.ndarray:
        resolved = layout or self.layout
        action = heuristic_action(obs, info, resolved, self.config)
        cell = resolved.player_features(obs).cell_fraction
        moving = int(action[0]) != 0
        enemies = resolved.perceived_enemies(obs)
        if enemies or not moving:
            # Combat footwork and deliberate standing still are not "stuck".
            self._waiting = 0
            self._rotation = 0
        elif cell == self._last_cell:
            self._waiting += 1
        else:
            self._waiting = 0
            self._rotation = max(self._rotation - 1, 0)
        self._last_cell = cell
        if self._rotation:
            action = action.copy()
            action[0] = _rotate_move(int(action[0]), self._rotation)
        if self._waiting >= self.no_progress_windows:
            self._rotation = (self._rotation + 1) % 8
            self._waiting = 0
        return action


def _rotate_move(move: int, steps: int) -> int:
    """Rotate a move index (1–8, clockwise in the published order) by ``steps``."""
    if move <= 0:
        return 0
    return ((int(move) - 1 + int(steps)) % 8) + 1


__all__ = [
    "DEFAULT_CONFIG",
    "HeuristicConfig",
    "HeuristicPolicy",
    "heuristic_action",
]
