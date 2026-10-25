"""Global navigation map built from the published observation (RL-4 round 3).

The ``ashcrown-obs-v1`` vector only publishes a 21x21 window of the terrain and
a 900 px perception radius, so a purely local policy cannot path to an enemy
across the map: on the measured level-1 seeds the last two enemies sit ~850 px
away *behind walls*, and a local BFS in the window can find neither the target
cell nor the corridor that leads to it (the RL-4 runs spent 233 of 300 game
seconds walking in circles around them).

The portal is the one globally published landmark (``docs/07`` §5.2 features
21–24, §5.7), so this module anchors a map on it:

* coordinates are **portal-relative cells**, ``m = cell − portal_cell``,
  integrated incrementally: the observation publishes the *continuous* portal
  distance and the player's sub-cell offset, so ``Δtile = −Δ(portal_rel) −
  Δ(player_cell_offset)`` per decision is exact (both terms are published and
  the true step is an integer). Integrating is what keeps the frame exact —
  rounding the raw ``portal_rel`` instead flips by a cell whenever the body
  crosses a tile boundary, which would smear every wall into two rows;
* every decision the 21x21 window is folded into two global rasters —
  *walkable* (floor, no wall, no solid prop: plan truth) and *visited* (the
  env's own per-floor memory, channel 5);
* :meth:`NavMap.step_towards` answers "the first grid step toward X" with four
  fallbacks — the target cell itself, the reachable frontier cell closest to
  the target (this is what *expands* the map toward the goal), the reachable
  cell closest to the target, and ``None`` (no idea → the caller keeps its
  local behaviour).

Everything here is a pure function of published observations plus the map the
policy itself built; there is no engine change and no schema change.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from .spaces import Layout, screen_delta

#: The eight legal iso-grid steps (clockwise, ``docs/07`` §6.1).
STEPS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)


def _screen_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Screen-space distance between two portal-relative grid cells."""
    delta = (float(a[0] - b[0]), float(a[1] - b[1]))
    return float(np.hypot(*screen_delta(delta)))


class NavMap:
    """Portal-relative walkable/visited raster with a grid BFS on top."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        #: Cells known to be floor without a wall or solid prop.
        self.walkable: set[tuple[int, int]] = set()
        #: Cells the episode has already visited (the env's own memory).
        self.visited: set[tuple[int, int]] = set()
        self.doors: set[tuple[int, int]] = set()
        #: Cells seen and known to be impassable (wall, solid prop, off-map).
        self.blocked: set[tuple[int, int]] = set()
        #: Known doorway cells: the only way out of a room, so an unreachable
        #: goal is routed through the doorway that points at it first.
        #: The player's portal-relative cell (updated by :meth:`update`).
        self.player: tuple[int, int] = (0, 0)
        self.updates = 0
        self._last_rel: tuple[float, float] | None = None
        self._last_offset: tuple[float, float] = (0.0, 0.0)
        #: Exact (float) map position of the player. The true value is always an
        #: integer, so accumulating the *exact* deltas and rounding once per
        #: window keeps the frame exact — rounding each step instead loses a
        #: cell every time the body crosses a tile boundary near the ±0.5 edge,
        #: and a one-cell fork splits the raster into two disconnected copies of
        #: the same room (measured: a third of the floor became "unreachable").
        self._exact: list[float] = [0.0, 0.0]
        #: The portal's own portal-relative cell (the map origin), published for
        #: callers that want to path *to* the exit.
        self.portal: tuple[int, int] = (0, 0)

    # -- learning the map --------------------------------------------------- #
    def update(
        self,
        grid: np.ndarray,
        portal: object,
        cell_offset: Sequence[float] = (0.0, 0.0),
    ) -> tuple[int, int]:
        """Fold the current 21x21 window into the global rasters.

        ``grid`` is ``Layout.grid(obs)``, ``portal`` a
        :class:`~ashcrown_rl.spaces.PortalFeatures` and ``cell_offset`` the
        ``info["player_cell_offset"]`` sub-cell diagnostic. Returns the player's
        portal-relative tile.
        """
        radius = self.layout.grid_radius
        size = self.layout.grid_size
        rel = (
            float(getattr(portal, "rel_cells", (0.0, 0.0))[0]),
            float(getattr(portal, "rel_cells", (0.0, 0.0))[1]),
        )
        offset = (float(cell_offset[0]), float(cell_offset[1]))
        if self._last_rel is None:
            self.player = (0, 0)
            # The frame origin is the player's first tile, so the portal's map
            # cell is fixed from the first observation onward.
            self.portal = (int(round(rel[0])), int(round(rel[1])))
        else:
            # Δtile = Δq − Δoffset, with Δq = −Δ(portal_rel) (the portal is
            # static) and offset = q − tile (the sub-cell diagnostic).
            step = (
                (self._last_rel[0] - rel[0]) - (offset[0] - self._last_offset[0]),
                (self._last_rel[1] - rel[1]) - (offset[1] - self._last_offset[1]),
            )
            if abs(step[0]) > 5.0 or abs(step[1]) > 5.0:
                # A new episode (the floor was rebuilt): restart the frame.
                self._exact = [0.0, 0.0]
            else:
                self._exact[0] += step[0]
                self._exact[1] += step[1]
            self.player = (int(round(self._exact[0])), int(round(self._exact[1])))
        self._last_rel = rel
        self._last_offset = offset
        anchor = (-self.player[0], -self.player[1])
        floor = self.layout.channel(grid, "floor").reshape(size, size)
        wall = self.layout.channel(grid, "wall").reshape(size, size)
        seen = self.layout.channel(grid, "visited").reshape(size, size)
        doors = self.layout.channel(grid, "doorway").reshape(size, size)
        # ``solid_prop`` is deliberately **not** part of the mask. A prop cell is
        # one the body can walk around at sub-cell resolution, but counting it as
        # a wall pinches 2-cell doorways shut: measured on seeds 2/3, five prop
        # cells cut a 28-cell room off the map, so the two surviving enemies were
        # "perceived but unreachable" for the whole episode. Long-range routing
        # ignores props; the local 21x21 BFS the combat core uses still avoids
        # them, and the pocket escape recovers any bump.
        walk = (floor >= 0.5) & (wall < 0.5)
        # Every window cell is recorded, including the walls: a "blocked" set
        # that only held walked-next-to walls would leave the rest of the wall
        # ring looking *unknown*, and the optimistic planner would then route
        # straight through it.
        rows, columns = np.mgrid[0:size, 0:size]
        for row, column in zip(rows.ravel().tolist(), columns.ravel().tolist(), strict=True):
            cell = (column - radius - anchor[0], row - radius - anchor[1])
            if walk[row, column]:
                self.walkable.add(cell)
            else:
                self.blocked.add(cell)
            if seen[row, column] >= 0.5:
                self.visited.add(cell)
            if doors[row, column] >= 0.5:
                self.doors.add(cell)
        self.updates += 1
        return self.player

    # -- querying the map --------------------------------------------------- #
    def frontier(self) -> set[tuple[int, int]]:
        """Known-walkable cells that were never visited."""
        return self.walkable - self.visited

    def bfs(self, start: Sequence[int] | None = None) -> dict[tuple[int, int], tuple[int, int]]:
        """BFS over the known-walkable cells: cell → ``(first step, depth)``.

        Diagonal steps need both orthogonal neighbours walkable, exactly like
        the local window BFS (``baseline._step_is_free``), so a step chosen here
        is always legal in the window the core sees.
        """
        origin = (
            (int(self.player[0]), int(self.player[1]))
            if start is None
            else (
                int(start[0]),
                int(start[1]),
            )
        )
        paths: dict[tuple[int, int], tuple[int, int]] = {origin: ((0, 0), 0)}
        if origin not in self.walkable:
            # The body stands on a cell the plan calls blocked (a prop overlap or
            # an unknown border cell): grow the search from its walkable
            # neighbours instead of giving up.
            for step in STEPS:
                neighbour = (origin[0] + step[0], origin[1] + step[1])
                if neighbour in self.walkable:
                    paths[neighbour] = (step, 1)
            queue: deque[tuple[int, int]] = deque(cell for cell in paths if cell != origin)
        else:
            queue = deque([origin])
        while queue:
            cell = queue.popleft()
            first_step, depth = paths[cell]
            for step in STEPS:
                nxt = (cell[0] + step[0], cell[1] + step[1])
                if nxt in paths or nxt not in self.walkable:
                    continue
                if step[0] != 0 and step[1] != 0:
                    if (cell[0] + step[0], cell[1]) not in self.walkable:
                        continue
                    if (cell[0], cell[1] + step[1]) not in self.walkable:
                        continue
                paths[nxt] = (step if cell == origin else first_step, depth + 1)
                queue.append(nxt)
        return paths

    def step_towards(self, target: Sequence[int]) -> tuple[int, int] | None:
        """First grid step toward ``target`` (portal-relative cells), or ``None``.

        Tiers, in order:

        1. the target cell, when it is known-walkable and reachable;
        2. the reachable **doorway** closest to the target — the only way out
           of the room the body is standing in (without this tier the policy
           parks in a corner of the room while the goal sits in the next one:
           measured on seeds 2/3/6, 233 s of standing still with two enemies
           alive and 100 HP);
        3. the reachable **frontier** cell closest to the target — walking it
           unfolds more of the floor *in the direction of the goal*, which is
           how the map eventually discovers the corridor to a far enemy;
        4. the reachable cell closest to the target, so the body keeps moving
           when the goal is walled off.
        """
        goal = (int(round(float(target[0]))), int(round(float(target[1]))))
        if goal == self.player:
            return None
        paths = self.bfs()
        reachable = [(cell, first, depth) for cell, (first, depth) in paths.items()]
        if goal in paths:
            return paths[goal][0]
        optimistic = self._optimistic_step(goal)
        if optimistic is not None:
            return optimistic
        if not reachable:
            return None
        doorways = [item for item in reachable if item[0] in self.doors]
        if doorways:
            _, first, _ = min(
                doorways,
                key=lambda item: (
                    round(_screen_distance(item[0], goal), 6),
                    item[2],
                    item[0][0],
                    item[0][1],
                ),
            )
            return first
        frontier = [item for item in reachable if item[0] not in self.visited]
        if frontier:
            _, first, _ = min(
                frontier,
                key=lambda item: (
                    round(_screen_distance(item[0], goal), 6),
                    item[2],
                    item[0][0],
                    item[0][1],
                ),
            )
            return first
        _, first, _ = min(
            reachable,
            key=lambda item: (
                round(_screen_distance(item[0], goal), 6),
                item[2],
                item[0][0],
                item[0][1],
            ),
        )
        return first

    def _optimistic_step(self, goal: tuple[int, int]) -> tuple[int, int] | None:
        """First step of a route that treats *unknown* cells as traversable.

        A cell the player could step into is always inside the current 21x21
        window, so "unknown" only ever means *far* terrain. Planning through it
        is therefore safe and is what unblocks the measured trap: the body stands
        in a room whose exits were never inside any window, every reachable
        known cell is already visited, and every tier that requires knowledge
        returns nothing (233 s of walking in circles with 2 enemies alive). The
        route is re-planned every window, so a wrong guess costs one blocked
        step and the next plan routes around the wall it just discovered.
        """
        if goal in self.blocked:
            return None
        cells = list(self.walkable) + [self.player, goal]
        x0 = min(cell[0] for cell in cells) - 2
        x1 = max(cell[0] for cell in cells) + 2
        y0 = min(cell[1] for cell in cells) - 2
        y1 = max(cell[1] for cell in cells) + 2
        start = self.player
        paths: dict[tuple[int, int], tuple[int, int]] = {start: (0, 0)}
        queue: deque[tuple[int, int]] = deque([start])
        best: tuple[int, int] | None = None
        best_distance = _screen_distance(start, goal)
        while queue:
            cell = queue.popleft()
            first = paths[cell][0]
            distance = _screen_distance(cell, goal)
            if distance < best_distance - 1e-9 or best is None:
                best_distance = distance
                best = first if cell != start else None
            if cell == goal:
                return first
            for step in STEPS:
                nxt = (cell[0] + step[0], cell[1] + step[1])
                if nxt in paths or nxt in self.blocked:
                    continue
                if not (x0 <= nxt[0] <= x1 and y0 <= nxt[1] <= y1):
                    continue
                paths[nxt] = (step if cell == start else first, 0)
                queue.append(nxt)
        return best

    def deepest_cell(self) -> tuple[int, int] | None:
        """The farthest reachable cell (the exploration destination)."""
        paths = self.bfs()
        reachable = [(cell, depth) for cell, (_first, depth) in paths.items()]
        if len(reachable) <= 1:
            return None
        cell, _ = max(reachable, key=lambda item: (item[1], item[0][0], item[0][1]))
        return cell

    def explore_step(self, goal: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """First step of the exploration walk when nothing is in sight.

        Tiers: the nearest unvisited known-walkable cell (clean up the local
        area), and otherwise the **farthest reachable** one. Walking to the far
        end of the known area is what unfolds the floor: the body sees 21x21
        around itself, so every arrival reveals new terrain at that end, and the
        next window's frontier is one step further. The RL-4 exploration stalled
        the other way round — it walked *towards the portal* whenever the local
        window was fully visited, and once the body stood on the inactive portal
        every further target rounded to its own cell.
        """
        paths = self.bfs()
        reachable = [(cell, first, depth) for cell, (first, depth) in paths.items()]
        if not reachable:
            return None
        frontier = [item for item in reachable if item[0] not in self.visited]
        if frontier:
            _, first, _ = min(frontier, key=lambda item: (item[2], item[0][0], item[0][1]))
            return first
        if goal is not None and goal in paths:
            return paths[goal][0]
        _, first, _ = max(reachable, key=lambda item: (item[2], item[0][0], item[0][1]))
        return first

    def frontier_step(self) -> tuple[int, int] | None:
        """First step to the nearest unvisited known-walkable cell."""
        paths = self.bfs()
        frontier = [
            (cell, first, depth)
            for cell, (first, depth) in paths.items()
            if cell not in self.visited and cell != self.player
        ]
        if not frontier:
            return None
        _, first, _ = min(
            frontier,
            key=lambda item: (item[2], item[0][0], item[0][1]),
        )
        return first


__all__ = ["NavMap", "STEPS"]
