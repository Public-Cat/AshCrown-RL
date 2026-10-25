"""Shared fixtures for the RL-4 tests.

* ``layout`` — the published ``ashcrown-obs-v1`` layout table, loaded from the
  stored handshake fixture (``tests/data/layout.json``), so the layout/action/
  baseline unit tests need no Godot process. ``test_spaces.py`` pins the fixture
  against the live handshake when the game is available.
* ``make_obs`` — builds a synthetic observation vector from natural-language
  values (player hp, enemy slots, portal delta, grid cells) using the layout's
  published offsets only.
* ``needs_godot`` — skips a test cleanly when the godot binary or the game
  checkout is missing (the same guard ``test_protocol.py`` uses).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from ashcrown_rl.launch import default_game_project, find_godot
from ashcrown_rl.spaces import Layout, load_fixture_layout


@pytest.fixture(scope="session")
def layout() -> Layout:
    """The published observation layout (from the stored handshake fixture)."""
    return load_fixture_layout()


class ObservationBuilder:
    """Build a valid ``ashcrown-obs-v1`` observation with named values."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.obs = np.zeros(layout.obs_len, dtype=np.float32)
        self.set("player", "alive", 1.0)
        self.set("player", "hp_frac", 1.0)
        self.set("player", "hp_abs", 100.0 / 400.0)
        self.set("player", "max_hp", 100.0 / 400.0)
        self.set("player", "attack_reach", 43.0 / 100.0)
        self.set("player", "move_speed", 220.0 / 500.0)
        self.set("player", "state_idle", 1.0)
        self.set("global", "level", 0.1)
        self.set("global", "mode_level", 1.0)

    # -- generic setters ---------------------------------------------------- #
    def set(self, block: str, name: str, value: float) -> ObservationBuilder:
        """Write one named feature (already normalised)."""
        self.obs[self.layout.offset(block, name)] = float(value)
        return self

    def set_raw(self, block: str, name: str, value: float) -> ObservationBuilder:
        """Write one feature from game units (the layout's denominator applies)."""
        denominator = 1.0
        from ashcrown_rl.spaces import DENOMINATORS

        denominator = DENOMINATORS.get(block, {}).get(name, 1.0)
        self.obs[self.layout.offset(block, name)] = float(value) / denominator
        return self

    # -- semantic helpers --------------------------------------------------- #
    def player(
        self,
        *,
        hp: float | None = None,
        facing: tuple[float, float] = (1.0, 0.0),
        state: str = "idle",
        attack_progress: float = 0.0,
        attack_reach_px: float = 43.0,
        velocity_px: tuple[float, float] = (0.0, 0.0),
    ) -> ObservationBuilder:
        if hp is not None:
            self.set_raw("player", "hp_abs", hp)
            self.set("player", "hp_frac", min(hp / 100.0, 1.0))
        self.set("player", "facing_cos", facing[0])
        self.set("player", "facing_sin", facing[1])
        for name in ("idle", "walk", "attack", "dead"):
            self.set("player", f"state_{name}", 1.0 if name == state else 0.0)
        self.set("player", "attack_progress", attack_progress)
        self.set_raw("player", "attack_reach", attack_reach_px)
        self.set_raw("player", "velocity_x", velocity_px[0])
        self.set_raw("player", "velocity_y", velocity_px[1])
        return self

    def enemy(
        self,
        slot: int,
        *,
        rel_cells: tuple[float, float],
        state: str = "chase",
        hp_frac: float = 1.0,
        kind: str = "skeleton_normal",
        attack_reach_px: float = 50.0,
        cooldown_s: float = 0.0,
    ) -> ObservationBuilder:
        block = self.layout.block("enemies")
        base = block.offset + slot * block.shape[1]
        index = {name: i for i, name in enumerate(block.fields)}
        row = np.zeros(block.shape[1], dtype=np.float32)
        row[index["present"]] = 1.0
        type_key = f"type_{kind}"
        if type_key in index:
            row[index[type_key]] = 1.0
        row[index["rel_x"]] = rel_cells[0] / self.layout.rel_cell_scale
        row[index["rel_y"]] = rel_cells[1] / self.layout.rel_cell_scale
        row[index["distance"]] = math.hypot(*rel_cells) / self.layout.rel_cell_scale
        screen = (rel_cells[0] - rel_cells[1], (rel_cells[0] + rel_cells[1]) * 0.5)
        length = math.hypot(*screen) or 1.0
        row[index["bearing_cos"]] = screen[0] / length
        row[index["bearing_sin"]] = screen[1] / length
        row[index["hp_frac"]] = hp_frac
        state_key = f"state_{state}"
        if state_key in index:
            row[index[state_key]] = 1.0
        row[index["attack_reach"]] = attack_reach_px / 200.0
        row[index["attack_cooldown"]] = cooldown_s / 5.0
        self.obs[base : base + block.shape[1]] = row
        return self

    def portal(self, rel_cells: tuple[float, float], *, active: bool = False) -> ObservationBuilder:
        self.set("global", "portal_active", 1.0 if active else 0.0)
        self.set_raw("global", "portal_rel_x", rel_cells[0])
        self.set_raw("global", "portal_rel_y", rel_cells[1])
        self.set_raw("global", "portal_distance", math.hypot(*rel_cells))
        self.set("global", "level_cleared", 1.0 if active else 0.0)
        return self

    def cleared(self, cleared: bool = True) -> ObservationBuilder:
        self.set("global", "level_cleared", 1.0 if cleared else 0.0)
        return self

    def enemies_alive(self, count: int) -> ObservationBuilder:
        self.set_raw("global", "enemies_alive", count)
        return self

    # -- terrain ------------------------------------------------------------ #
    def fill_floor(self, only: Iterable[tuple[int, int]] | None = None) -> ObservationBuilder:
        """Mark the whole local window as floor (minus explicitly blocked cells)."""
        radius = self.layout.grid_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if only is not None and (dx, dy) not in set(only):
                    continue
                self.grid(dx, dy, "floor", 1.0)
        return self

    def grid(self, dx: int, dy: int, channel: str, value: float = 1.0) -> ObservationBuilder:
        size = self.layout.grid_size
        block = self.layout.block("grid")
        channel_index = self.layout.grid_channels.index(channel)
        cell = self.layout.grid_index(dx, dy)
        self.obs[block.offset + channel_index * size * size + cell] = float(value)
        return self

    def build(self) -> np.ndarray:
        return np.clip(self.obs, -1.0, 1.0)


@pytest.fixture()
def make_obs(layout: Layout) -> Any:
    """Factory fixture: ``make_obs().player(...).build()``."""

    def _factory() -> ObservationBuilder:
        return ObservationBuilder(layout)

    return _factory


def godot_available() -> bool:
    """True when both the godot binary and the game checkout exist."""
    return find_godot() is not None and default_game_project() is not None


@pytest.fixture(scope="session")
def needs_godot() -> None:
    """Skip the test when a real headless process cannot be launched."""
    if not godot_available():
        pytest.skip("godot binary or game checkout unavailable")


def launch_kwargs(project: str | None = None) -> dict[str, Any]:
    """``AshCrownEnv`` kwargs that point at the local game checkout."""
    game = default_game_project()
    godot = find_godot()
    if game is None or godot is None:  # pragma: no cover - guarded by needs_godot
        pytest.skip("godot binary or game checkout unavailable")
    return {"godot_bin": godot, "project": project or str(game)}


@pytest.fixture(scope="session")
def env_kwargs(needs_godot: None) -> dict[str, Any]:
    """Session-wide ``AshCrownEnv`` kwargs for the game checkout."""
    del needs_godot
    return launch_kwargs()
