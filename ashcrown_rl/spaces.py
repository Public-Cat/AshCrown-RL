"""Observation layout slicing, action table and observation helpers (RL-4).

The engine publishes the authoritative ``ashcrown-obs-v1`` layout in the
handshake (game spec ``docs/07`` §5.1, §8.2): block names, offsets, sizes,
shapes, feature names, the enemy/pickup tables, the action directions and the
grid geometry. :class:`Layout` parses that table once and exposes **named
slices** — no observation offset is hardcoded anywhere in this package.

Two normalisation facts come from the spec (not the layout table):

* relative positions/distances are iso **grid cells** divided by
  ``rel_cell_scale`` (published, 32), and
* screen-pixel quantities (velocity, reach, damage) keep their pixel
  denominators (``docs/07`` §5.2–§5.5, mirrored in :data:`DENOMINATORS`).

The module also owns the semantic action table (:class:`ActionSpec`) derived
from the published ``action_directions``: screen vector, iso-grid step and the
best-matching move index for a direction.

Everything here is pure: no process, no I/O, so the heuristic baseline and the
tests can run on a stored layout fixture (``tests/data/layout.json``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Block names of ``ashcrown-obs-v1`` (``docs/07`` §5.1).
GLOBAL = "global"
PLAYER = "player"
ENEMIES = "enemies"
PICKUPS = "pickups"
HAZARDS = "hazards"
MERCHANT = "merchant"
EVENTS = "events"
GRID = "grid"

#: Nominal screen pixels per iso grid step (``FloorPlan.CELL`` in the game).
#: Only used to convert a pixel reach/velocity into grid-cell units; the
#: observation never carries absolute positions, so nothing else needs it.
PX_PER_CELL = 128.0

#: Denominator of every normalised feature, per block (``docs/07`` §5.2–§5.5).
#: ``to_raw`` multiplies by it; helpers below use it for the features they
#: expose as physical units.
DENOMINATORS: Mapping[str, Mapping[str, float]] = {
    GLOBAL: {
        "level": 10.0,
        "game_seconds": 600.0,
        "enemies_spawned": 30.0,
        "enemies_alive": 30.0,
        "kills": 100.0,
        "gold": 500.0,
        "gem_red": 50.0,
        "gem_blue": 50.0,
        "gem_green": 50.0,
        "gem_white": 50.0,
        "boss_aoe_timer": 10.0,
        "portal_rel_x": 32.0,
        "portal_rel_y": 32.0,
        "portal_distance": 32.0,
        "merchant_rel_x": 32.0,
        "merchant_rel_y": 32.0,
        "merchant_distance": 32.0,
        "nearest_enemy_distance": 32.0,
    },
    PLAYER: {
        "hp_abs": 400.0,
        "max_hp": 400.0,
        "damage": 80.0,
        "move_speed": 500.0,
        "attack_speed_mult": 3.0,
        "attack_reach": 100.0,
        "attack_speed_bonus": 0.5,
        "velocity_x": 500.0,
        "velocity_y": 500.0,
    },
    ENEMIES: {
        "rel_x": 32.0,
        "rel_y": 32.0,
        "distance": 32.0,
        "move_speed": 500.0,
        "damage": 80.0,
        "windup_remaining": 3.0,
        "attack_cooldown": 5.0,
        "attack_reach": 200.0,
        "scale": 2.0,
    },
    PICKUPS: {"rel_x": 32.0, "rel_y": 32.0, "distance": 32.0},
    HAZARDS: {"rel_x": 32.0, "rel_y": 32.0, "radius": 200.0, "remaining": 2.0},
    MERCHANT: {"rel_x": 32.0, "rel_y": 32.0, "distance": 32.0},
    EVENTS: {
        "damage_dealt": 100.0,
        "damage_taken": 100.0,
        "kills": 5.0,
        "gold_gained": 10.0,
        "gems_gained": 2.0,
        "attacks_started": 5.0,
        "hits_landed": 5.0,
        "player_healed": 50.0,
    },
}


class LayoutError(ValueError):
    """The published layout table is inconsistent or incomplete."""


# --------------------------------------------------------------------------- #
# Layout table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    """One published observation block (flat, or shaped like an enemy slot)."""

    name: str
    offset: int
    size: int
    shape: tuple[int, ...]
    fields: tuple[str, ...]
    channels: tuple[str, ...] = ()

    @property
    def end(self) -> int:
        """Offset of the first float after this block."""
        return self.offset + self.size

    def slice(self, obs: np.ndarray) -> np.ndarray:
        """Flat view of the block (no copy for a float32 array)."""
        return np.asarray(obs, dtype=np.float32)[self.offset : self.end]

    def array(self, obs: np.ndarray) -> np.ndarray:
        """The block reshaped to its published shape (e.g. ``(12, 39)``)."""
        return self.slice(obs).reshape(self.shape)

    def field_index(self, name: str) -> int:
        """Block-local index of a named feature (``0`` for the first)."""
        try:
            return self.fields.index(name)
        except ValueError as exc:
            raise LayoutError(f"block {self.name!r} has no field {name!r}") from exc

    def __repr__(self) -> str:  # pragma: no cover - dataclass repr is too long
        return f"Block({self.name!r}, offset={self.offset}, shape={self.shape})"


@dataclass(frozen=True)
class ActionSpec:
    """One movement action (``docs/07`` §6.1): screen vector + iso grid step."""

    index: int
    screen: tuple[float, float]
    grid: tuple[int, int]

    @property
    def moving(self) -> bool:
        """False for action ``0`` (no movement)."""
        return self.grid != (0, 0)

    def describe(self) -> str:
        """Human-readable meaning for ``metadata["action_meanings"]``."""
        if not self.moving:
            return "0:stay"
        return f"{self.index}:screen={self.screen},grid={self.grid}"


@dataclass(frozen=True)
class Layout:
    """Parsed ``ashcrown-obs-v1`` layout (built from the handshake)."""

    obs_len: int
    blocks: tuple[Block, ...]
    action_directions: tuple[tuple[float, float], ...]
    enemy_types: tuple[str, ...]
    enemy_states: tuple[str, ...]
    pickup_kinds: tuple[str, ...]
    grid_radius: int
    grid_size: int
    grid_channels: tuple[str, ...]
    rel_cell_scale: float
    perception_radius: float
    protocol: int = 0
    obs_schema: str = ""
    action_schema: str = ""
    layout_hash: str = ""
    _blocks: dict[str, Block] = field(default_factory=dict, repr=False)
    _actions: tuple[ActionSpec, ...] = field(default=(), repr=False)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_handshake(cls, hello: Mapping[str, Any]) -> Layout:
        """Build the layout from a verified handshake payload."""
        layout = hello.get("layout")
        if not isinstance(layout, Mapping) or not isinstance(layout.get("blocks"), list):
            raise LayoutError("handshake carries no layout block table")
        layout_hash = hello.get("layout_hash", "")
        obs_len = hello.get("obs_len")
        if isinstance(obs_len, int) and int(layout.get("obs_len", obs_len)) != obs_len:
            raise LayoutError(f"layout obs_len={layout.get('obs_len')!r} != handshake {obs_len}")
        return cls.from_layout_dict(layout, layout_hash=str(layout_hash))

    @classmethod
    def from_layout_dict(cls, layout: Mapping[str, Any], *, layout_hash: str = "") -> Layout:
        """Build the layout from the published ``layout`` object (or a fixture)."""
        raw_blocks = layout.get("blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise LayoutError("layout has no blocks")
        blocks: list[Block] = []
        offset = 0
        for entry in raw_blocks:
            if not isinstance(entry, Mapping):
                raise LayoutError("layout block is not an object")
            name = str(entry.get("name", ""))
            size = entry.get("size")
            if not name or not isinstance(size, int) or size <= 0:
                raise LayoutError(f"layout block {name!r} has an invalid size {size!r}")
            if int(entry.get("offset", offset)) != offset:
                raise LayoutError(
                    f"layout block {name!r} starts at {entry.get('offset')!r}, expected {offset}"
                )
            fields_raw = entry.get("fields") or []
            fields = tuple(str(value) for value in fields_raw)
            shape_raw = entry.get("shape")
            shape = tuple(int(value) for value in shape_raw) if shape_raw else (size,)
            if int(np.prod(shape)) != size:
                raise LayoutError(f"layout block {name!r} shape {shape} covers {size} floats")
            if fields and len(fields) != int(np.prod(shape[1:]) if len(shape) > 1 else size):
                raise LayoutError(f"layout block {name!r} has {len(fields)} fields for {shape}")
            blocks.append(
                Block(
                    name=name,
                    offset=offset,
                    size=size,
                    shape=shape,
                    fields=fields,
                    channels=tuple(str(value) for value in entry.get("channels") or ()),
                )
            )
            offset += size
        declared = layout.get("obs_len")
        if declared is not None and int(declared) != offset:
            raise LayoutError(f"layout blocks cover {offset} floats but obs_len is {declared}")
        directions = layout.get("action_directions")
        if not isinstance(directions, list) or len(directions) < 9:
            raise LayoutError("layout carries no 9-entry action direction table")
        grid = layout.get("grid") or {}
        channels = blocks[-1].channels if blocks[-1].name == GRID else ()
        instance = cls(
            obs_len=offset,
            blocks=tuple(blocks),
            action_directions=tuple((float(pair[0]), float(pair[1])) for pair in directions),
            enemy_types=tuple(str(value) for value in layout.get("enemy_types") or ()),
            enemy_states=tuple(str(value) for value in layout.get("enemy_states") or ()),
            pickup_kinds=tuple(str(value) for value in layout.get("pickup_kinds") or ()),
            grid_radius=int(grid.get("radius", 0)),
            grid_size=int(grid.get("size", 0)),
            grid_channels=tuple(str(value) for value in grid.get("channel_names") or channels),
            rel_cell_scale=float(layout.get("rel_cell_scale", 32.0)),
            perception_radius=float(layout.get("perception_radius", 0.0)),
            protocol=int(layout.get("protocol", 0)),
            obs_schema=str(layout.get("obs_schema", "")),
            action_schema=str(layout.get("action_schema", "")),
            layout_hash=layout_hash,
        )
        object.__setattr__(instance, "_blocks", {block.name: block for block in blocks})
        object.__setattr__(instance, "_actions", instance._build_actions())
        instance._validate()
        return instance

    def _validate(self) -> None:
        for required in (GLOBAL, PLAYER, ENEMIES, EVENTS):
            if required not in self._blocks:
                raise LayoutError(f"layout is missing the {required!r} block")
        for name in (ENEMIES, PICKUPS, HAZARDS):
            block = self._blocks.get(name)
            if block is not None and len(block.shape) != 2:
                raise LayoutError(f"layout block {name!r} is not a slot matrix")
        enemy = self._blocks[ENEMIES]
        if self.enemy_types and len(enemy.fields) != enemy.shape[1]:
            raise LayoutError("enemy slot fields do not match the published shape")
        if GRID in self._blocks:
            grid = self._blocks[GRID]
            expected = (len(self.grid_channels), self.grid_size, self.grid_size)
            if grid.shape != expected:
                raise LayoutError(f"grid block shape {grid.shape} != {expected}")

    def _build_actions(self) -> tuple[ActionSpec, ...]:
        """Screen vector and iso grid step of every action (derived, not literal)."""
        actions: list[ActionSpec] = []
        for index, (sx, sy) in enumerate(self.action_directions):
            if sx == 0.0 and sy == 0.0:
                actions.append(ActionSpec(index=index, screen=(0.0, 0.0), grid=(0, 0)))
                continue
            # Exact inverse of IsoMath.to_screen: grid = (2*sy + sx, 2*sy - sx),
            # normalised so the largest component is one grid step.
            gx = 2.0 * sy + sx
            gy = 2.0 * sy - sx
            scale = max(abs(gx), abs(gy))
            actions.append(
                ActionSpec(
                    index=index,
                    screen=(sx, sy),
                    grid=(round(gx / scale), round(gy / scale)),
                )
            )
        return tuple(actions)

    # -- table access ------------------------------------------------------- #
    def block(self, name: str) -> Block:
        """The named block, or :class:`LayoutError`."""
        try:
            return self._blocks[name]
        except KeyError as exc:
            raise LayoutError(f"no {name!r} block in the layout") from exc

    def has_block(self, name: str) -> bool:
        """True when the handshake published this block."""
        return name in self._blocks

    def fields(self, name: str) -> tuple[str, ...]:
        """Feature names of a block (empty for the grid)."""
        return self.block(name).fields

    def index(self, name: str, field_name: str) -> int:
        """Block-local index of a feature."""
        return self.block(name).field_index(field_name)

    def offset(self, name: str, field_name: str) -> int:
        """Absolute observation index of a feature."""
        block = self.block(name)
        return block.offset + block.field_index(field_name)

    def value(self, obs: np.ndarray, name: str, field_name: str) -> float:
        """One named feature as a Python float."""
        return float(np.asarray(obs, dtype=np.float32)[self.offset(name, field_name)])

    def get(self, obs: np.ndarray, dotted: str) -> float:
        """``layout.get(obs, "global.portal_distance")``."""
        block_name, _, field_name = dotted.partition(".")
        if not field_name:
            raise LayoutError(f"'{dotted}' is not a block.field name")
        return self.value(obs, block_name, field_name)

    def to_raw(self, name: str, field_name: str, value: float) -> float:
        """Denormalise one feature back to game units."""
        denominator = DENOMINATORS.get(name, {}).get(field_name, 1.0)
        return float(value) * float(denominator)

    # -- named slices ------------------------------------------------------- #
    def slice(self, obs: np.ndarray, name: str) -> np.ndarray:
        """Flat view of a block."""
        return self.block(name).slice(obs)

    def matrix(self, obs: np.ndarray, name: str) -> np.ndarray:
        """Shaped view of a slot block (``(K, slot_size)``)."""
        return self.block(name).array(obs)

    def grid(self, obs: np.ndarray) -> np.ndarray:
        """Local terrain grid as ``(channels, size, size)``."""
        return self.block(GRID).array(obs)

    def slices(self) -> dict[str, list[int]]:
        """``{block: [offset, size]}`` — the compact layout metadata in ``info``."""
        return {block.name: [block.offset, block.size] for block in self.blocks}

    # -- action table ------------------------------------------------------- #
    @property
    def actions(self) -> tuple[ActionSpec, ...]:
        """The nine movement actions in published order."""
        return self._actions

    def action_table(self) -> tuple[ActionSpec, ...]:
        """Alias of :attr:`actions` for callers that prefer a verb."""
        return self._actions

    def action_meanings(self) -> list[str]:
        """``metadata["action_meanings"]`` (``move``, ``attack``, ``interact``)."""
        meanings = [spec.describe() for spec in self._actions]
        meanings.append("attack: pulse (0/1), fires on the first tick of the window")
        meanings.append("interact: pulse (0/1), merchant only")
        return meanings

    def move_index(self, screen_delta: Sequence[float]) -> int:
        """Index of the action whose screen direction best matches a delta.

        Returns ``0`` (stay) for a zero delta.
        """
        dx = float(screen_delta[0])
        dy = float(screen_delta[1])
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            return 0
        best_index = 0
        best_dot = -np.inf
        for spec in self._actions:
            if not spec.moving:
                continue
            dot = (spec.screen[0] * dx + spec.screen[1] * dy) / length
            if dot > best_dot + 1e-9:
                best_dot = dot
                best_index = spec.index
        return best_index

    def move_for_grid_step(self, step: Sequence[int]) -> int | None:
        """Action index of an iso grid step, or ``None`` when it is not a move."""
        wanted = (int(round(step[0])), int(round(step[1])))
        for spec in self._actions:
            if spec.grid == wanted:
                return spec.index
        return None

    def opposite_move(self, index: int) -> int:
        """The action that moves exactly opposite to ``index`` (``0`` stays ``0``)."""
        spec = self._actions[int(index)] if 0 <= int(index) < len(self._actions) else None
        if spec is None or not spec.moving:
            return 0
        opposite = (-spec.grid[0], -spec.grid[1])
        found = self.move_for_grid_step(opposite)
        return 0 if found is None else found

    # -- grid helpers ------------------------------------------------------- #
    def grid_index(self, dx: int, dy: int) -> int:
        """Index inside one grid channel of the local cell ``(dx, dy)``.

        ``dx, dy`` are cell deltas relative to the player's cell, each in
        ``[-radius, radius]`` (``docs/07`` §5.6, channel-major, row-major).
        """
        radius = self.grid_radius
        size = self.grid_size
        if abs(dx) > radius or abs(dy) > radius:
            raise LayoutError(f"cell delta ({dx}, {dy}) is outside the {size}x{size} grid")
        return (int(dy) + radius) * size + (int(dx) + radius)

    def channel(self, grid: np.ndarray, name: str) -> np.ndarray:
        """One grid channel by published name (``floor``, ``wall``, …)."""
        try:
            index = self.grid_channels.index(name)
        except ValueError as exc:
            raise LayoutError(f"no grid channel {name!r}") from exc
        return np.asarray(grid, dtype=np.float32)[index]

    def cell(self, grid: np.ndarray, dx: int, dy: int, channel: str) -> float:
        """Value of one local cell in one channel."""
        return float(self.channel(grid, channel).reshape(-1)[self.grid_index(dx, dy)])

    def walkable(self, grid: np.ndarray, dx: int, dy: int) -> bool:
        """True when the local cell is floor, not wall and not a solid prop."""
        if abs(dx) > self.grid_radius or abs(dy) > self.grid_radius:
            return False
        if self.cell(grid, dx, dy, "floor") <= 0.5:
            return False
        if self.cell(grid, dx, dy, "wall") >= 0.5:
            return False
        return self.cell(grid, dx, dy, "solid_prop") < 0.5

    def visited(self, grid: np.ndarray, dx: int, dy: int) -> float:
        """``visited`` channel of a local cell (env-maintained memory)."""
        return self.cell(grid, dx, dy, "visited")

    def hazard(self, grid: np.ndarray, dx: int, dy: int) -> float:
        """``hazard`` channel of a local cell (live telegraph coverage)."""
        return self.cell(grid, dx, dy, "hazard")

    # -- semantic accessors ------------------------------------------------- #
    def player_features(self, obs: np.ndarray) -> PlayerFeatures:
        """Player block as physical units (``docs/07`` §5.3)."""
        block = self.slice(obs, PLAYER)
        index = {name: i for i, name in enumerate(self.block(PLAYER).fields)}
        states = ("idle", "walk", "attack", "dead")
        return PlayerFeatures(
            alive=bool(block[index["alive"]] >= 0.5),
            hp_frac=float(block[index["hp_frac"]]),
            hp=float(block[index["hp_abs"]]) * 400.0,
            max_hp=float(block[index["max_hp"]]) * 400.0,
            armor_fraction=float(block[index["armor"]]),
            damage=float(block[index["damage"]]) * 80.0,
            move_speed_px=float(block[index["move_speed"]]) * 500.0,
            attack_speed_mult=float(block[index["attack_speed_mult"]]) * 3.0,
            cell_fraction=(float(block[index["cell_x"]]), float(block[index["cell_y"]])),
            facing=(float(block[index["facing_cos"]]), float(block[index["facing_sin"]])),
            state=next((name for name in states if block[index[f"state_{name}"]] >= 0.5), "other"),
            attack_progress=float(block[index["attack_progress"]]),
            attack_hitbox_active=bool(block[index["attack_hitbox_active"]] >= 0.5),
            attack_queued=bool(block[index["attack_queued"]] >= 0.5),
            knocked_back=bool(block[index["knocked_back"]] >= 0.5),
            attack_reach_px=float(block[index["attack_reach"]]) * 100.0,
            velocity_px=(
                float(block[index["velocity_x"]]) * 500.0,
                float(block[index["velocity_y"]]) * 500.0,
            ),
        )

    def enemy_slots(self, obs: np.ndarray) -> np.ndarray:
        """The ``(K, slot_size)`` enemy matrix."""
        return self.matrix(obs, ENEMIES)

    def perceived_enemies(self, obs: np.ndarray) -> list[EnemySlot]:
        """Present, living enemy slots in published (nearest-first) order."""
        slots = self.enemy_slots(obs)
        block = self.block(ENEMIES)
        index = {name: i for i, name in enumerate(block.fields)}
        state_names = self.enemy_states
        found: list[EnemySlot] = []
        for slot in range(slots.shape[0]):
            row = slots[slot]
            if row[index["present"]] < 0.5:
                continue
            type_name = ""
            for candidate in self.enemy_types:
                key = f"type_{candidate}"
                if key in index and row[index[key]] >= 0.5:
                    type_name = candidate
                    break
            state_name = "other"
            for candidate in state_names:
                key = f"state_{candidate}"
                if key in index and row[index[key]] >= 0.5:
                    state_name = candidate
                    break
            rel = (
                float(row[index["rel_x"]]) * self.rel_cell_scale,
                float(row[index["rel_y"]]) * self.rel_cell_scale,
            )
            found.append(
                EnemySlot(
                    slot=slot,
                    type_name=type_name,
                    rel_cells=rel,
                    distance_cells=float(row[index["distance"]]) * self.rel_cell_scale,
                    bearing=(float(row[index["bearing_cos"]]), float(row[index["bearing_sin"]])),
                    hp_frac=float(row[index["hp_frac"]]),
                    state=state_name,
                    state_progress=float(row[index["state_progress"]]),
                    windup_remaining_s=float(row[index["windup_remaining"]]) * 3.0,
                    hitbox_active=bool(row[index["attack_hitbox_active"]] >= 0.5),
                    attack_cooldown_s=float(row[index["attack_cooldown"]]) * 5.0,
                    is_elite=bool(row[index["is_elite"]] >= 0.5),
                    is_boss=bool(row[index["is_boss"]] >= 0.5),
                    attack_reach_px=float(row[index["attack_reach"]]) * 200.0,
                    asleep=bool(row[index["asleep"]] >= 0.5),
                )
            )
        return found

    def nearest_enemy(self, obs: np.ndarray) -> EnemySlot | None:
        """The nearest perceived living enemy, or ``None``."""
        enemies = self.perceived_enemies(obs)
        return enemies[0] if enemies else None

    def portal(self, obs: np.ndarray) -> PortalFeatures:
        """Portal block from the global summary (``docs/07`` §5.2 features 21–24)."""
        active = self.value(obs, GLOBAL, "portal_active") >= 0.5
        rel = (
            self.value(obs, GLOBAL, "portal_rel_x") * self.rel_cell_scale,
            self.value(obs, GLOBAL, "portal_rel_y") * self.rel_cell_scale,
        )
        distance = self.value(obs, GLOBAL, "portal_distance") * self.rel_cell_scale
        screen = screen_delta(rel)
        return PortalFeatures(
            active=active,
            rel_cells=rel,
            distance_cells=distance,
            screen_px=screen * PX_PER_CELL,
            bearing=normalise(screen),
        )

    def merchant(self, obs: np.ndarray) -> MerchantFeatures:
        """Merchant block (``docs/07`` §5.5)."""
        present = self.value(obs, MERCHANT, "present") >= 0.5
        rel = (
            self.value(obs, MERCHANT, "rel_x") * self.rel_cell_scale,
            self.value(obs, MERCHANT, "rel_y") * self.rel_cell_scale,
        )
        return MerchantFeatures(
            present=present,
            rel_cells=rel,
            distance_cells=self.value(obs, MERCHANT, "distance") * self.rel_cell_scale,
            within_range=self.value(obs, MERCHANT, "within_interact_range") >= 0.5,
        )

    def pickups(self, obs: np.ndarray) -> list[PickupSlot]:
        """Present pickup slots in published (nearest-first) order."""
        slots = self.matrix(obs, PICKUPS)
        block = self.block(PICKUPS)
        index = {name: i for i, name in enumerate(block.fields)}
        found: list[PickupSlot] = []
        for slot in range(slots.shape[0]):
            row = slots[slot]
            if row[index["present"]] < 0.5:
                continue
            kind = ""
            for candidate in self.pickup_kinds:
                key = f"kind_{candidate}"
                if key in index and row[index[key]] >= 0.5:
                    kind = candidate
                    break
            found.append(
                PickupSlot(
                    slot=slot,
                    kind=kind,
                    rel_cells=(
                        float(row[index["rel_x"]]) * self.rel_cell_scale,
                        float(row[index["rel_y"]]) * self.rel_cell_scale,
                    ),
                    distance_cells=float(row[index["distance"]]) * self.rel_cell_scale,
                )
            )
        return found

    def events(self, obs: np.ndarray) -> Events:
        """Per-window event deltas in game units (``docs/07`` §5.8)."""
        raw = self.slice(obs, EVENTS)
        values = {name: float(raw[i]) for i, name in enumerate(self.block(EVENTS).fields)}
        return Events(
            damage_dealt=values.get("damage_dealt", 0.0) * 100.0,
            damage_taken=values.get("damage_taken", 0.0) * 100.0,
            kills=int(round(values.get("kills", 0.0) * 5.0)),
            attacks_started=int(round(values.get("attacks_started", 0.0) * 5.0)),
            hits_landed=int(round(values.get("hits_landed", 0.0) * 5.0)),
            level_cleared=values.get("level_cleared", 0.0) >= 0.5,
            died=values.get("died", 0.0) >= 0.5,
            portal_entered=values.get("portal_entered", 0.0) >= 0.5,
        )

    # -- scalar summaries --------------------------------------------------- #
    def level(self, obs: np.ndarray) -> int:
        """Level index (0.1–1.0 normalised in the vector)."""
        return int(round(self.value(obs, GLOBAL, "level") * 10.0))

    def game_seconds(self, obs: np.ndarray) -> float:
        """Episode game time in seconds."""
        return self.value(obs, GLOBAL, "game_seconds") * 600.0

    def tick_fraction(self, obs: np.ndarray) -> float:
        """``episode_tick / max_ticks`` (0–1)."""
        return self.value(obs, GLOBAL, "tick")

    def enemies_alive(self, obs: np.ndarray) -> int:
        """Living enemy count for the floor (0–30 normalised)."""
        return int(round(self.value(obs, GLOBAL, "enemies_alive") * 30.0))

    def enemies_spawned(self, obs: np.ndarray) -> int:
        """Enemies spawned this floor (kills + alive)."""
        return int(round(self.value(obs, GLOBAL, "enemies_spawned") * 30.0))

    def kills(self, obs: np.ndarray) -> int:
        """Kills this run (0–100 normalised)."""
        return int(round(self.value(obs, GLOBAL, "kills") * 100.0))

    def level_cleared(self, obs: np.ndarray) -> bool:
        """True once the floor is cleared (``level_cleared``)."""
        return self.value(obs, GLOBAL, "level_cleared") >= 0.5

    def portal_active(self, obs: np.ndarray) -> bool:
        """True when the exit portal is usable."""
        return self.value(obs, GLOBAL, "portal_active") >= 0.5

    def wave_active(self, obs: np.ndarray) -> bool:
        """True while a wave is running (arena levels)."""
        return self.value(obs, GLOBAL, "wave_active") >= 0.5

    def nearest_enemy_bearing(self, obs: np.ndarray) -> tuple[float, float]:
        """Screen bearing of the nearest perceived enemy as ``(cos, sin)``."""
        return (
            self.value(obs, GLOBAL, "nearest_enemy_bearing_cos"),
            self.value(obs, GLOBAL, "nearest_enemy_bearing_sin"),
        )

    def nearest_enemy_distance_cells(self, obs: np.ndarray) -> float:
        """Distance to the nearest perceived enemy in iso cells."""
        return self.value(obs, GLOBAL, "nearest_enemy_distance") * self.rel_cell_scale


# --------------------------------------------------------------------------- #
# Semantic value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlayerFeatures:
    """Player state in game units (``docs/07`` §5.3)."""

    alive: bool
    hp_frac: float
    hp: float
    max_hp: float
    armor_fraction: float
    damage: float
    move_speed_px: float
    attack_speed_mult: float
    cell_fraction: tuple[float, float]
    facing: tuple[float, float]
    state: str
    attack_progress: float
    attack_hitbox_active: bool
    attack_queued: bool
    knocked_back: bool
    attack_reach_px: float
    velocity_px: tuple[float, float]

    @property
    def healthy(self) -> bool:
        """True while the player is alive with positive HP."""
        return self.alive and self.hp > 0.0


@dataclass(frozen=True)
class EnemySlot:
    """One perceived enemy (``docs/07`` §5.4) in game units."""

    slot: int
    type_name: str
    rel_cells: tuple[float, float]
    distance_cells: float
    bearing: tuple[float, float]
    hp_frac: float
    state: str
    state_progress: float
    windup_remaining_s: float
    hitbox_active: bool
    attack_cooldown_s: float
    is_elite: bool
    is_boss: bool
    attack_reach_px: float
    asleep: bool

    @property
    def screen_delta_cells(self) -> np.ndarray:
        """Screen image of the grid delta, in grid-cell units."""
        return screen_delta(self.rel_cells)

    @property
    def screen_distance_px(self) -> float:
        """Screen-space distance in pixels."""
        return float(np.hypot(*self.screen_delta_cells)) * PX_PER_CELL

    @property
    def winding_up(self) -> bool:
        """True while the enemy is telegraphing a swing."""
        return self.state == "windup"

    @property
    def attacking(self) -> bool:
        """True while the enemy's hitbox is live."""
        return self.state == "attack" or self.hitbox_active

    @property
    def dead(self) -> bool:
        """True once the slot reports the dead state."""
        return self.state == "dead" or self.hp_frac <= 0.0


@dataclass(frozen=True)
class PortalFeatures:
    """Portal state from the global summary."""

    active: bool
    rel_cells: tuple[float, float]
    distance_cells: float
    screen_px: np.ndarray
    bearing: np.ndarray


@dataclass(frozen=True)
class MerchantFeatures:
    """Merchant summary (``docs/07`` §5.5)."""

    present: bool
    rel_cells: tuple[float, float]
    distance_cells: float
    within_range: bool


@dataclass(frozen=True)
class PickupSlot:
    """One pickup slot."""

    slot: int
    kind: str
    rel_cells: tuple[float, float]
    distance_cells: float


@dataclass(frozen=True)
class Events:
    """Per-window event deltas (``docs/07`` §5.8) in game units."""

    damage_dealt: float
    damage_taken: float
    kills: int
    attacks_started: int
    hits_landed: int
    level_cleared: bool
    died: bool
    portal_entered: bool


# --------------------------------------------------------------------------- #
# Iso helpers (the single conversion used by every bearing in this package)
# --------------------------------------------------------------------------- #
def screen_delta(cells: Sequence[float]) -> np.ndarray:
    """Screen image of an iso grid delta: ``(x - y, (x + y) / 2)``."""
    x = float(cells[0])
    y = float(cells[1])
    return np.array([x - y, (x + y) * 0.5], dtype=np.float64)


def grid_delta(screen: Sequence[float]) -> np.ndarray:
    """Iso grid delta of a screen vector (exact inverse of :func:`screen_delta`)."""
    sx = float(screen[0])
    sy = float(screen[1])
    return np.array([sx * 0.5 + sy, sy - sx * 0.5], dtype=np.float64)


def normalise(vector: Sequence[float]) -> np.ndarray:
    """Unit vector of ``vector`` (a zero vector stays zero)."""
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    length = float(np.hypot(*array)) if array.size == 2 else float(np.linalg.norm(array))
    if length <= 1e-9:
        return np.zeros_like(array)
    return array / length


def fixture_path() -> Path:
    """Path of the stored layout fixture (``tests/data/layout.json``)."""
    return Path(__file__).resolve().parents[1] / "tests" / "data" / "layout.json"


def load_fixture_layout(path: Path | str | None = None) -> Layout:
    """Load the stored layout fixture as a :class:`Layout` (no process needed)."""
    target = Path(path) if path is not None else fixture_path()
    with target.open(encoding="utf-8") as handle:
        layout = json.load(handle)
    return Layout.from_layout_dict(layout)


def dump_layout_fixture(layout: Mapping[str, Any], path: Path | str | None = None) -> Path:
    """Write a handshake ``layout`` object to the fixture path (maintenance)."""
    target = Path(path) if path is not None else fixture_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DENOMINATORS",
    "ENEMIES",
    "EVENTS",
    "GLOBAL",
    "GRID",
    "HAZARDS",
    "MERCHANT",
    "PICKUPS",
    "PLAYER",
    "PX_PER_CELL",
    "ActionSpec",
    "Block",
    "EnemySlot",
    "Events",
    "Layout",
    "LayoutError",
    "MerchantFeatures",
    "PickupSlot",
    "PlayerFeatures",
    "PortalFeatures",
    "dump_layout_fixture",
    "fixture_path",
    "grid_delta",
    "load_fixture_layout",
    "normalise",
    "screen_delta",
]
