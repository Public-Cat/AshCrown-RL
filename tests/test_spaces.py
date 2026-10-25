"""Unit tests for the layout/action/observation helpers (RL-4, ``docs/07`` §5–§6).

Everything here runs on the stored handshake layout fixture: no Godot process.
The fixture itself is verified against the live handshake by the integration
test at the bottom (which skips cleanly without a game checkout).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ashcrown_rl.launch import GodotProcess, LaunchConfig
from ashcrown_rl.protocol import load_compat
from ashcrown_rl.spaces import (
    ENEMIES,
    GLOBAL,
    HAZARDS,
    PX_PER_CELL,
    Block,
    Layout,
    LayoutError,
    fixture_path,
    grid_delta,
    load_fixture_layout,
    normalise,
    screen_delta,
)


def test_fixture_parses_the_published_table(layout: Layout) -> None:
    assert layout.obs_len == 3254
    assert [block.name for block in layout.blocks] == [
        "global",
        "player",
        "enemies",
        "pickups",
        "hazards",
        "merchant",
        "events",
        "grid",
    ]
    offsets = 0
    for block in layout.blocks:
        assert block.offset == offsets, block.name
        offsets += block.size
    assert offsets == layout.obs_len
    assert layout.block(ENEMIES).shape == (12, 39)
    assert layout.block("grid").shape == (6, 21, 21)
    assert layout.grid_radius == 10
    assert layout.rel_cell_scale == 32.0


def test_named_slices_index_by_name_not_offset(layout: Layout) -> None:
    obs = np.zeros(layout.obs_len, dtype=np.float32)
    obs[layout.offset(GLOBAL, "portal_distance")] = 0.5
    assert layout.value(obs, GLOBAL, "portal_distance") == pytest.approx(0.5)
    assert layout.get(obs, "global.portal_distance") == pytest.approx(0.5)
    with pytest.raises(LayoutError):
        layout.value(obs, GLOBAL, "does_not_exist")
    with pytest.raises(LayoutError):
        layout.block("nope")


def test_layout_rejects_an_inconsistent_table(layout: Layout) -> None:
    raw = {
        "blocks": [
            {"name": "global", "offset": 3, "size": 31, "fields": []},
        ],
        "obs_len": 31,
    }
    with pytest.raises(LayoutError, match="starts at"):
        Layout.from_layout_dict(raw)
    with pytest.raises(LayoutError, match="no blocks"):
        Layout.from_layout_dict({"blocks": []})


def test_action_table_is_derived_from_the_published_directions(layout: Layout) -> None:
    """The grid step of every move comes from the screen vector (no literals)."""
    expected = {
        1: (1, -1),
        2: (1, 0),
        3: (1, 1),
        4: (0, 1),
        5: (-1, 1),
        6: (-1, 0),
        7: (-1, -1),
        8: (0, -1),
    }
    assert layout.actions[0].grid == (0, 0)
    for index, step in expected.items():
        assert layout.actions[index].grid == step, index
        assert layout.move_for_grid_step(step) == index
    # Screen order is clockwise starting at east.
    assert layout.move_index((1.0, 0.0)) == 1
    assert layout.move_index((0.0, 1.0)) == 3
    assert layout.move_index((-1.0, 0.0)) == 5
    assert layout.move_index((0.0, -1.0)) == 7
    assert layout.move_index((0.0, 0.0)) == 0
    # Opposite steps round-trip.
    for index in range(1, 9):
        opposite = layout.opposite_move(index)
        assert (
            layout.move_for_grid_step(
                (-layout.actions[index].grid[0], -layout.actions[index].grid[1])
            )
            == opposite
        )


def test_iso_helpers_round_trip() -> None:
    assert np.allclose(screen_delta((1, 0)), (1.0, 0.5))
    assert np.allclose(grid_delta(screen_delta((3, -2))), (3.0, -2.0))
    assert np.allclose(normalise((0.0, 0.0)), (0.0, 0.0))
    assert float(np.hypot(*normalise((3.0, 4.0)))) == pytest.approx(1.0)


def test_player_features_are_in_game_units(layout: Layout, make_obs) -> None:
    obs = make_obs().player(hp=72.0, velocity_px=(220.0, 0.0), attack_reach_px=43.0).build()
    player = layout.player_features(obs)
    assert player.hp == pytest.approx(72.0)
    assert player.max_hp == pytest.approx(100.0)
    assert player.hp_frac == pytest.approx(0.72)
    assert player.attack_reach_px == pytest.approx(43.0)
    assert player.move_speed_px == pytest.approx(220.0)
    assert player.velocity_px == pytest.approx((220.0, 0.0))
    assert player.state == "idle"
    assert player.alive


def test_enemy_slots_are_sorted_and_typed(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .enemy(0, rel_cells=(4.0, 0.0), kind="skeleton_normal")
        .enemy(1, rel_cells=(2.0, 0.0), kind="slime_green", hp_frac=0.5)
        .build()
    )
    enemies = layout.perceived_enemies(obs)
    assert [enemy.type_name for enemy in enemies] == ["skeleton_normal", "slime_green"]
    nearest = layout.nearest_enemy(obs)
    assert nearest is not None
    assert nearest.slot == 0
    assert nearest.distance_cells == pytest.approx(4.0)
    # rel (4, 0) → screen (4, 2) cells → 2 * sqrt(5) * 128 px
    assert nearest.screen_distance_px == pytest.approx(4.47213595 * PX_PER_CELL, rel=1e-3)
    assert nearest.bearing == pytest.approx(tuple(normalise((4.0, 2.0))), rel=1e-6)
    assert nearest.dead is False
    assert enemies[1].hp_frac == pytest.approx(0.5)


def test_nearest_enemy_is_none_when_nothing_is_perceived(layout: Layout, make_obs) -> None:
    assert layout.nearest_enemy(make_obs().build()) is None
    assert layout.perceived_enemies(make_obs().build()) == []


def test_portal_summary_exposes_distance_and_bearing(layout: Layout, make_obs) -> None:
    obs = make_obs().portal((0.0, 8.0), active=True).build()
    portal = layout.portal(obs)
    assert portal.active
    assert portal.distance_cells == pytest.approx(8.0)
    # grid (0, 8) → screen (-8, 4) cells.
    assert portal.screen_px == pytest.approx(np.array([-8.0, 4.0]) * PX_PER_CELL)
    assert layout.portal_active(obs)
    assert layout.level_cleared(obs)


def test_grid_lookups_use_the_published_geometry(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .grid(1, 0, "wall")
        .grid(0, 1, "solid_prop")
        .grid(-1, 0, "doorway")
        .grid(1, 0, "floor", 0.0)
        .grid(0, -1, "visited")
        .build()
    )
    grid = layout.grid(obs)
    assert layout.walkable(grid, 0, 0)
    assert not layout.walkable(grid, 1, 0), "a wall cell is never walkable"
    assert not layout.walkable(grid, 0, 1), "a solid prop blocks the cell"
    assert layout.cell(grid, -1, 0, "doorway") == 1.0
    assert layout.visited(grid, 0, -1) == 1.0
    assert not layout.walkable(grid, 99, 0), "outside the window is not walkable"
    with pytest.raises(LayoutError):
        layout.grid_index(99, 0)
    with pytest.raises(LayoutError):
        layout.channel(grid, "nope")


def test_block_slice_returns_a_view(layout: Layout) -> None:
    obs = np.arange(layout.obs_len, dtype=np.float32)
    block = layout.block(GLOBAL)
    assert isinstance(block, Block)
    assert block.slice(obs)[0] == obs[0]
    matrix = layout.enemy_slots(obs)
    assert matrix.shape == (12, 39)
    enemies_offset = layout.block(ENEMIES).offset
    assert matrix[0, 0] == obs[enemies_offset]
    assert matrix[1, 0] == obs[enemies_offset + 39]


def test_fixture_is_the_layout_the_compat_file_pins(layout: Layout) -> None:
    """The stored layout is the one ``compat.json`` pins (guards fixture drift)."""
    pinned = load_compat()["layout_hash"]
    assert len(pinned) == 64
    assert layout.obs_len == 3254
    assert layout.obs_schema == load_compat()["obs_schema"]
    assert layout.action_schema == load_compat()["action_schema"]
    assert layout.protocol == load_compat()["protocol"]


@pytest.mark.integration
def test_live_handshake_matches_the_stored_layout_fixture(env_kwargs: dict) -> None:
    """The fixture must be exactly what the engine publishes (no drift)."""
    config = LaunchConfig(game_project=env_kwargs["project"], godot_bin=env_kwargs["godot_bin"])
    with GodotProcess(config) as process:
        client = process.connect()
        assert client.hello is not None
        live = Layout.from_handshake(client.hello)
        client.close()
    fixture = load_fixture_layout()
    assert fixture.blocks == live.blocks
    assert fixture.action_directions == live.action_directions
    assert fixture.enemy_types == live.enemy_types
    assert fixture.grid_channels == live.grid_channels
    assert fixture.obs_len == live.obs_len


# --------------------------------------------------------------------------- #
# v2: the layout is a function of the observation config (RL-6)
# --------------------------------------------------------------------------- #
def _fixture_dict() -> dict:
    return json.loads(fixture_path().read_text(encoding="utf-8"))


def _no_grid_layout() -> Layout:
    """The table the engine publishes for ``obs_grid=false`` (608 floats)."""
    raw = _fixture_dict()
    raw["blocks"] = [block for block in raw["blocks"] if block["name"] != "grid"]
    raw["obs_len"] = 608
    raw["observation_config"] = {**raw["observation_config"], "obs_grid": False}
    raw["grid"] = {
        "enabled": False,
        "radius": 0,
        "size": 0,
        "channels": 0,
        "channel_names": [],
    }
    return Layout.from_layout_dict(raw)


def _radius_layout(radius: int) -> Layout:
    """A copy of the fixture table with a resized grid block."""
    raw = _fixture_dict()
    size = radius * 2 + 1
    channels = len(raw["grid"]["channel_names"])
    grid = raw["blocks"][-1]
    assert grid["name"] == "grid"
    grid["shape"] = [channels, size, size]
    grid["size"] = channels * size * size
    raw["obs_len"] += grid["size"] - 6 * 21 * 21
    raw["grid"]["radius"] = radius
    raw["grid"]["size"] = size
    raw["observation_config"] = {**raw["observation_config"], "obs_grid_radius": radius}
    return Layout.from_layout_dict(raw)


def test_layout_without_grid_has_no_slice_and_no_grid_geometry() -> None:
    layout = _no_grid_layout()
    assert layout.obs_len == 608
    assert layout.grid_enabled is False
    assert layout.grid_radius == 0
    assert layout.grid_size == 0
    assert not layout.has_block("grid")
    with pytest.raises(LayoutError, match="no 'grid' block"):
        layout.block("grid")
    obs = np.zeros(layout.obs_len, dtype=np.float32)
    with pytest.raises(LayoutError, match="obs_grid=false"):
        layout.grid(obs)


def test_layout_radius_changes_the_grid_shape_and_indexing() -> None:
    for radius in (4, 21):
        layout = _radius_layout(radius)
        size = radius * 2 + 1
        assert layout.grid_enabled is True
        assert layout.grid_radius == radius
        assert layout.grid_size == size
        assert layout.block("grid").shape == (6, size, size)
        assert layout.obs_len == 608 + 6 * size * size
        assert layout.observation_config["obs_grid_radius"] == radius
        obs = np.zeros(layout.obs_len, dtype=np.float32)
        obs[layout.offset(GLOBAL, "level")] = 0.3
        grid = layout.grid(obs)
        assert grid.shape == (6, size, size)
        # centre cell of the window
        assert layout.grid_index(0, 0) == radius * size + radius
        with pytest.raises(LayoutError, match="outside"):
            layout.grid_index(radius + 1, 0)
        assert layout.cell(grid, 0, 0, "floor") == 0.0


def test_layout_rejects_a_grid_that_disagrees_with_its_config() -> None:
    raw = _fixture_dict()
    raw["observation_config"] = {**raw["observation_config"], "obs_grid": False}
    with pytest.raises(LayoutError, match="obs_grid is false"):
        Layout.from_layout_dict(raw)
    raw_dict = _fixture_dict()
    raw_dict["blocks"] = [b for b in raw_dict["blocks"] if b["name"] != "grid"]
    raw_dict["obs_len"] = 608
    raw_dict["observation_config"] = {**raw_dict["observation_config"], "obs_grid": True}
    with pytest.raises(LayoutError, match="no grid block"):
        Layout.from_layout_dict(raw_dict)


def test_hazard_slot_publishes_the_v2_telegraph_features(layout: Layout) -> None:
    fields = layout.block(HAZARDS).fields
    assert fields[:6] == (
        "present",
        "rel_x",
        "rel_y",
        "radius",
        "remaining",
        "kind",
    ), "v1's six features keep their order"
    assert fields[6:] == ("telegraph_fraction", "coverage")
    assert layout.block(HAZARDS).shape == (4, 8)
    obs = np.zeros(layout.obs_len, dtype=np.float32)
    hazards = layout.matrix(obs, HAZARDS)
    hazards[0, layout.index(HAZARDS, "telegraph_fraction")] = 0.25
    hazards[0, layout.index(HAZARDS, "coverage")] = 0.5
    assert layout.matrix(obs, HAZARDS)[0, 6] == pytest.approx(0.25)
    assert layout.matrix(obs, HAZARDS)[0, 7] == pytest.approx(0.5)
