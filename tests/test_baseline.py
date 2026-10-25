"""Heuristic baseline tests (RL-4, ``docs/07`` §9.4).

The decision rules are unit-tested on synthetic observations built from the
published layout (no process). The ``integration`` tests drive the real level 1:
the fixed-seed **acceptance** gate (``docs/07`` §9.4: the shipped configuration
must clear at least 8 of the 10 fixed seeds), a progress smoke test, a
determinism test and the seed-10 pocket regression. The acceptance test costs a
couple of minutes; the per-seed detail behind it is the committed artifact
``docs/artifacts/RL/baseline_level1.json`` / ``.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ashcrown_rl.baseline import (
    DEFAULT_CONFIG,
    ENEMY_TIMING,
    HeuristicConfig,
    HeuristicPolicy,
    _enemy_timing,
    _strafe_step,
    _swing_covers,
    _time_to_hit,
    frontier_open,
    heuristic_action,
)
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.navmap import NavMap
from ashcrown_rl.spaces import (
    PX_PER_CELL,
    Layout,
    PortalFeatures,
    grid_delta,
    load_fixture_layout,
    normalise,
    screen_delta,
)

#: The 10 fixed acceptance seeds (``docs/07`` §9.4).
ACCEPTANCE_SEEDS: tuple[int, ...] = tuple(range(1, 11))
#: Clears the shipped configuration must reach on those seeds.
ACCEPTANCE_CLEARS = 8


def _portal(rel_x: float, rel_y: float) -> PortalFeatures:
    """A portal feature block for the NavMap frame tests."""
    return PortalFeatures(
        active=False,
        rel_cells=(rel_x, rel_y),
        distance_cells=float(np.hypot(rel_x, rel_y)),
        screen_px=np.asarray(screen_delta((rel_x, rel_y))) * PX_PER_CELL,
        bearing=np.zeros(2),
    )


def info_with_offset(offset: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    """Minimal ``info`` carrying the sub-cell diagnostic the nav map needs."""
    return {"player_cell_offset": [offset[0], offset[1]]}


def _dot(action: np.ndarray, layout: Layout, direction: np.ndarray) -> float:
    spec = layout.actions[int(action[0])]
    unit = normalise(direction)
    return float(spec.screen[0] * unit[0] + spec.screen[1] * unit[1])


def _closes_in(layout: Layout, action: np.ndarray, target_cells: Any) -> bool:
    """True when the action's grid step gets closer to ``target_cells``."""
    step = layout.actions[int(action[0])].grid
    before = float(np.hypot(*screen_delta(target_cells)))
    after = float(np.hypot(*screen_delta((target_cells[0] - step[0], target_cells[1] - step[1]))))
    return after < before - 1e-9


def test_dead_player_does_nothing(layout: Layout, make_obs) -> None:
    obs = make_obs().set("player", "alive", 0.0).build()
    action = heuristic_action(obs, {}, layout)
    assert action.tolist() == [0, 0, 0]


def test_moves_toward_the_nearest_enemy(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().enemy(0, rel_cells=(4.0, 1.0)).player().build()
    action = heuristic_action(obs, {}, layout)
    enemy = layout.nearest_enemy(obs)
    assert enemy is not None
    assert action[1] == 0, "out of reach: no swing yet"
    assert _closes_in(layout, action, enemy.rel_cells)


def test_attacks_when_in_reach_and_facing(layout: Layout, make_obs) -> None:
    # 0.40 grid cells along +x → screen (0.40, 0.20) cells → 0.447 cells →
    # 57 px: inside the warrior's 43 + 20 px strike volume, with the enemy
    # chasing (so its own hitbox cannot be live this window).
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.40, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action.tolist() == [0, 1, 0]


def test_does_not_swing_from_outside_the_measured_strike_volume(layout: Layout, make_obs) -> None:
    """Measured on level 1: a swing started beyond ~64 px starts whiffing."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.60, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0, "walk in first"
    assert _closes_in(layout, action, (0.60, 0.0))


def test_approaches_when_it_is_not_facing_the_enemy(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.25, 0.0), state="chase")
        .player(facing=(-1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0, "a swing would lock the wrong facing"
    assert action[0] != 0, "it steps to turn instead"


def test_circles_the_attacker_during_the_windup(layout: Layout, make_obs) -> None:
    """Round 4: the answer to a telegraph is a *perpendicular* step, not a retreat.

    The enemy's attack volume is an oriented box that freezes with its facing, so
    backing away along the bearing stays inside it; a lateral step leaves it and
    keeps the body in punish range (``scripts/trace_episode.py``: 87 % of skeleton
    swings were still on the body at the active frames with the straight retreat,
    and the lateral margin at that moment has a median of 24 px).
    """
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.2, 0.0), state="windup", attack_reach_px=50.0)
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    enemy = layout.nearest_enemy(obs)
    assert enemy is not None
    assert action[1] == 0, "never trade into a telegraph"
    assert action[0] != 0, "it moves"
    assert abs(_dot(action, layout, enemy.screen_delta_cells)) < 0.75, "not along the facing"


def test_never_queues_a_second_swing(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.25, 0.0), state="chase")
        .player(facing=(1.0, 0.0), state="attack", attack_progress=0.5)
        .build()
    )
    assert heuristic_action(obs, {}, layout).tolist() == [0, 0, 0]


def test_follows_the_portal_once_cleared(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().portal((6.0, 2.0), active=True).player().build()
    action = heuristic_action(obs, {}, layout)
    portal = layout.portal(obs)
    assert action[1] == 0
    assert _closes_in(layout, action, portal.rel_cells)


def test_keeps_closing_onto_the_portal_trigger(layout: Layout, make_obs) -> None:
    """Inside the portal's own cell the policy still steers onto the trigger.

    Parking in the cell centre (the RL-4 behaviour) never overlaps the portal's
    ``Area2D``, so an otherwise-cleared seed ran out of time next to it. A grid
    step overshoots a sub-cell target, so the assertion is on the *direction*:
    the body walks 15 px per decision along it and crosses the trigger.
    """
    obs = make_obs().fill_floor().portal((0.3, -0.2), active=True).player().build()
    action = heuristic_action(obs, {}, layout)
    assert action[2] == 1, "pulses interact"
    assert action[0] != 0, "still walks the last fraction of a cell"
    assert _dot(action, layout, screen_delta((0.3, -0.2))) > 0.5, "the step points at the portal"


def test_brakes_when_already_on_the_portal(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().portal((0.0, 0.0), active=True).player().build()
    action = heuristic_action(obs, {}, layout)
    assert action[0] == 0, "nothing left to walk"
    assert action[2] == 1, "pulses interact"


def test_explores_an_unvisited_cell_without_targets(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().build()
    grid = layout.grid(obs)
    assert frontier_open(layout, grid)
    action = heuristic_action(obs, {}, layout)
    assert action[0] != 0
    step = layout.actions[int(action[0])].grid
    assert layout.walkable(grid, step[0], step[1])


def test_walks_around_a_wall_instead_of_into_it(layout: Layout, make_obs) -> None:
    """The step is chosen on walkable terrain even when the direct line is walled."""
    builder = make_obs().fill_floor().portal((5.0, 0.0), active=True)
    for dy in range(-layout.grid_radius, layout.grid_radius + 1):
        builder.grid(1, dy, "floor", 0.0).grid(1, dy, "wall", 1.0)
    obs = builder.build()
    grid = layout.grid(obs)
    action = heuristic_action(obs, {}, layout)
    step = layout.actions[int(action[0])].grid
    assert step != (1, 0), "the direct step is walled off"
    assert layout.walkable(grid, step[0], step[1])


def test_steps_off_a_live_telegraph(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().grid(0, 0, "hazard").portal((3.0, 3.0), active=True).build()
    action = heuristic_action(obs, {}, layout)
    step = layout.actions[int(action[0])].grid
    assert step != (0, 0)
    assert layout.hazard(layout.grid(obs), step[0], step[1]) < 0.5


def test_action_is_a_pure_function_of_the_observation(layout: Layout, make_obs) -> None:
    obs = make_obs().fill_floor().enemy(0, rel_cells=(2.0, 1.0)).build()
    first = heuristic_action(obs, {}, layout)
    second = heuristic_action(obs.copy(), {}, layout)
    assert first.tolist() == second.tolist()
    assert first.dtype == np.int64


def test_config_changes_the_decision(layout: Layout, make_obs) -> None:
    """``standoff_frac`` keeps the body out of point-blank trades by default."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.15, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    # ~21 px: inside the swing volume but far below the standoff band, so the
    # default backs out instead of trading into the enemy's next telegraph.
    default = heuristic_action(obs, {}, layout, DEFAULT_CONFIG)
    assert default[1] == 0, "never swing from point blank"
    assert default[0] != 0, "it steps back into the band"
    eager = heuristic_action(
        obs, {}, layout, HeuristicConfig(standoff_frac=0.0, standoff_min_px=0.0)
    )
    assert eager.tolist() == [0, 1, 0], "with the band disabled it swings"


def test_turns_onto_a_target_the_swing_would_miss(layout: Layout, make_obs) -> None:
    """Facing is the last movement direction, so a miss turns with one step."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.40, 0.0), state="chase")
        .player(facing=(0.0, 1.0))  # facing 90 degrees away from the enemy
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0, "the swing volume does not cover a target beside it"
    assert _dot(action, layout, screen_delta((0.40, 0.0))) > 0.9, "it turns onto it"


def test_anticipates_a_second_enemy_telegraph(layout: Layout, make_obs) -> None:
    """A non-nearest enemy that is telegraphing is dodged too."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(2.0, 0.0), state="chase")
        .enemy(1, rel_cells=(-0.15, 0.0), state="windup", attack_reach_px=50.0)
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0
    assert action[0] != 0


def test_policy_wrapper_breaks_a_loop(layout: Layout, make_obs) -> None:
    """The stateful wrapper rotates the heading when the cell never changes."""
    policy = HeuristicPolicy(layout, no_progress_windows=3, loop_window=8)
    obs = make_obs().fill_floor().build()
    info: dict[str, Any] = {"tick": 0, "action": [1, 0, 0]}
    actions = []
    for _ in range(6):
        actions.append(int(policy(obs, info, layout)[0]))
        info = {"tick": 4, "action": [actions[-1], 0, 0]}
    assert len(set(actions)) > 1, "the wrapper must not repeat a blocked heading"


@pytest.mark.integration
def test_heuristic_lands_kills_on_level_1(env_kwargs: dict[str, Any]) -> None:
    """Progress smoke test: the scripted policy converts observations into kills."""
    env = AshCrownEnv(level=1, max_ticks=18_000, **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        obs, info = env.reset(seed=2)
        kills = 0
        for _ in range(1500):
            obs, _, terminated, truncated, info = env.step(policy(obs, info, env.layout))
            kills = max(kills, int(info.get("kills", 0)))
            if terminated or truncated:
                break
        assert kills >= 1, "the heuristic never killed anything on level 1"
        assert info["state_hash"]
    finally:
        env.close()


@pytest.mark.integration
def test_acceptance_clears_eight_of_the_ten_fixed_seeds(env_kwargs: dict[str, Any]) -> None:
    """``docs/07`` §9.4 acceptance: >= 8 of the 10 fixed seeds clear level 1.

    The same protocol as the committed artifact (warrior, level 1,
    ``frame_skip=4``, ``max_ticks=18,000``, ``DEFAULT_CONFIG``). It costs a
    couple of minutes because it is the real gate, not a proxy: the RL-4 review
    found the report claiming 8/10 while the suite asserted only "kills happen".
    """
    env = AshCrownEnv(level=1, max_ticks=18_000, **env_kwargs)
    cleared: list[int] = []
    reasons: dict[int, str] = {}
    try:
        for seed in ACCEPTANCE_SEEDS:
            policy = HeuristicPolicy(env.layout)
            obs, info = env.reset(seed=seed)
            while not env.episode_done:
                obs, _, _, _, info = env.step(policy(obs, info, env.layout))
            reasons[seed] = str(info.get("terminal_reason", ""))
            if reasons[seed] == "level_clear":
                cleared.append(seed)
    finally:
        env.close()
    assert len(cleared) >= ACCEPTANCE_CLEARS, (
        f"only {len(cleared)}/{len(ACCEPTANCE_SEEDS)} fixed seeds cleared "
        f"({cleared}); terminal reasons {reasons}"
    )


@pytest.mark.integration
def test_heuristic_is_deterministic_for_a_seed(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        runs = []
        for _ in range(2):
            policy = HeuristicPolicy(env.layout)
            obs, info = env.reset(seed=4242)
            hashes = []
            while not env.episode_done:
                obs, _, _, _, info = env.step(policy(obs, info, env.layout))
                hashes.append(info["state_hash"])
            runs.append(hashes)
        assert runs[0] == runs[1]
        assert len(runs[0]) > 0
    finally:
        env.close()


def _engine_cell_px(project: str) -> float | None:
    """``FloorPlan.CELL`` from the game sources, or ``None`` when unavailable."""
    path = Path(project) / "scripts" / "levelgen" / "floor_plan.gd"
    if not path.is_file():
        return None
    match = re.search(r"^const CELL:\s*float\s*=\s*([0-9.]+)", path.read_text(), re.MULTILINE)
    return None if match is None else float(match.group(1))


@pytest.mark.integration
def test_reach_conversion_matches_the_engine_cell_size(env_kwargs: dict[str, Any]) -> None:
    """``PX_PER_CELL`` is the engine's ``FloorPlan.CELL``, pinned on a live reset.

    The pixel quantities the baseline reasons about (reaches, velocities) are
    converted into grid cells with :data:`PX_PER_CELL`; the engine places cells
    with ``FloorPlan.CELL``. The pin uses the two ``info`` keys the RL-4 follow-up
    added (``docs/07`` §5.6): ``player_position`` converted back through
    :func:`grid_delta` must equal ``player_cell`` shifted by the ground layer's
    half-tile origin, in cells.
    """
    cell_px = _engine_cell_px(env_kwargs["project"])
    env = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        _, info = env.reset(seed=20260113)
        cell = np.asarray(info["player_cell"], dtype=np.float64)
        position = np.asarray(info["player_position"], dtype=np.float64)
        assert not np.allclose(cell, 0.0), "the plan frame must be absolute"
        if cell_px is not None:
            assert PX_PER_CELL == pytest.approx(cell_px, rel=1e-9)
        assert np.allclose(
            grid_delta(position) / PX_PER_CELL, cell + np.array([1.0, 0.0]), atol=1e-3
        ), f"player_cell {cell} disagrees with player_position {position}"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Pocket escape (RL-4 follow-up, issue #94)
# --------------------------------------------------------------------------- #
def test_policy_escapes_a_pocket_when_the_engine_reports_stuck(layout: Layout, make_obs) -> None:
    """``info.stuck`` (docs/07 §5.7) short-circuits the progress counter."""
    policy = HeuristicPolicy(layout)
    obs = make_obs().fill_floor().portal((5.0, 1.0), active=True).player().build()
    core = heuristic_action(obs, {}, layout)
    info = {"action": [int(core[0]), 0, 0], "player_cell_offset": [0.2, -0.2], "stuck": True}
    escaped = int(policy(obs, info, layout)[0])
    assert escaped != int(core[0]), "a stuck body is turned onto another axis"
    step = layout.actions[escaped].grid
    assert layout.walkable(layout.grid(obs), step[0], step[1]), "the escape axis is free"


def test_policy_escapes_a_ping_pong_the_cell_never_changes(layout: Layout, make_obs) -> None:
    """A two-cell ping-pong keeps ``cell_fraction`` moving; the offset does not.

    The pocket the RL-4 baseline fell into on seeds 1 and 10: the body oscillated
    inside a couple of cells for the whole episode, so neither the cell index nor
    the velocity stuck detector saw it. The sub-cell odometry does.
    """
    policy = HeuristicPolicy(layout, loop_window=10)
    obs = make_obs().fill_floor().build()
    core = heuristic_action(obs, {}, layout)
    assert int(core[0]) != 0, "the core wants to walk"
    moves = []
    for step in range(14):
        offset = [0.3, 0.0] if step % 2 == 0 else [-0.3, 0.0]
        info = {"action": [int(core[0]), 0, 0], "player_cell_offset": offset}
        moves.append(int(policy(obs, info, layout)[0]))
    assert len(set(moves)) > 1, "walking without net progress must change the heading"


def test_policy_keeps_fighting_instead_of_breaking_off(layout: Layout, make_obs) -> None:
    """Combat footwork is not a pocket: no escape override while an enemy is close."""
    policy = HeuristicPolicy(layout)
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.0), state="windup", attack_reach_px=50.0, windup_s=0.1)
        .player(facing=(1.0, 0.0))
        .build()
    )
    core = heuristic_action(obs, {}, layout)
    assert core[1] == 0, "the telegraph is dodged, not traded"
    assert core[0] != 0, "the dodge is a real step"
    info = {"action": [int(core[0]), 0, 0], "player_cell_offset": [0.0, 0.0], "stuck": False}
    assert policy(obs, info, layout).tolist() == core.tolist()


@pytest.mark.integration
def test_seed_10_does_not_pocket_any_more(env_kwargs: dict[str, Any]) -> None:
    """Regression for the seeds 1/10 pocket: the sweep must leave the spawn room.

    Before the RL-4 follow-up these seeds covered only 10-24 cells in the whole
    5-game-minute episode (a two-cell ping-pong), so nothing was ever perceived.
    """
    env = AshCrownEnv(level=1, max_ticks=18_000, **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        obs, info = env.reset(seed=10)
        layout = env.layout
        cells: set[tuple[int, int]] = set()
        for _ in range(600):
            obs, _, terminated, truncated, info = env.step(policy(obs, info, layout))
            fraction = layout.player_features(obs).cell_fraction
            cells.add((round(fraction[0] * 34), round(fraction[1] * 34)))
            if terminated or truncated:
                break
        assert len(cells) >= 8, f"only {len(cells)} distinct cells visited"
        assert info["state_hash"], "the episode produced a state"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Round 3: measured combat economics (issue #94)
# --------------------------------------------------------------------------- #
def test_swing_volume_covers_the_measured_band() -> None:
    """A swing reaches ~65 px ahead and ~40 px to the side (engine geometry)."""
    assert _swing_covers((1.0, 0.0), (0.0, 0.0), 43.0), "an overlapping enemy is hit"
    assert _swing_covers((1.0, 0.0), (60.0, 0.0), 43.0), "60 px ahead is inside"
    assert not _swing_covers((1.0, 0.0), (75.0, 0.0), 43.0), "75 px ahead is outside"
    assert not _swing_covers((1.0, 0.0), (-45.0, 0.0), 43.0), "a target behind is missed"
    assert _swing_covers((1.0, 0.0), (20.0, 40.0), 43.0), "the box is wide (measured 100 %)"


def test_swing_volume_matches_the_measured_land_rate(layout: Layout, make_obs) -> None:
    """Every swing the trace recorded as landing must pass the volume test.

    ``scripts/trace_episode.py`` measured 100-128 % land rate over 0-70 px and
    every facing cosine, so the geometric gate must not refuse those.
    """
    for distance_px in (20.0, 40.0, 55.0, 65.0):
        for angle_deg in (0.0, 20.0, 40.0, 60.0):
            radians = np.radians(angle_deg)
            offset = (distance_px * np.cos(radians), distance_px * np.sin(radians))
            assert _swing_covers((1.0, 0.0), offset, 43.0), (distance_px, angle_deg)


def test_backs_out_of_the_standoff_band_before_swinging(layout: Layout, make_obs) -> None:
    """A point-blank swing cannot be escaped before the next telegraph."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.12, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0
    assert action[0] != 0, "it does not stand still at point blank"


def test_interrupts_a_telegraph_that_is_still_early(layout: Layout, make_obs) -> None:
    """A hit at +0.32 s cancels a 0.35 s telegraph, so an early windup is a swing."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.42, 0.0), state="windup", windup_s=0.34)
        .player(facing=(1.0, 0.0))
        .build()
    )
    assert heuristic_action(obs, {}, layout).tolist() == [0, 1, 0]


def test_dodges_a_telegraph_that_would_land_first(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.42, 0.0), state="windup", windup_s=0.10)
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0
    assert action[0] != 0, "it steps out of the frozen box"
    assert abs(_dot(action, layout, screen_delta((0.42, 0.0)))) < 0.75


def test_retreats_from_two_simultaneous_attacks(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.30, 0.0), state="windup", windup_s=0.05)
        .enemy(1, rel_cells=(-0.30, 0.0), state="windup", windup_s=0.10)
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action[1] == 0, "no single swing can interrupt two attackers"
    assert action[0] != 0, "it moves out"


def test_navmap_frame_tracks_the_body() -> None:
    """``Δtile = −Δportal_rel − Δplayer_cell_offset`` (the exact frame identity)."""
    layout = load_fixture_layout()
    nav = NavMap(layout)
    grid = np.zeros(
        (len(layout.grid_channels), layout.grid_size * layout.grid_size), dtype=np.float32
    )
    nav.update(grid, _portal(-4.0, 2.0), (0.0, 0.0))
    assert nav.player == (0, 0)
    assert nav.portal == (-4, 2)
    # Walk one cell east: the portal delta loses exactly one cell, the sub-cell
    # offset is unchanged (``offset = q − tile`` and both moved by one).
    nav.update(grid, _portal(-5.0, 2.0), (0.0, 0.0))
    assert nav.player == (1, 0)
    # Sub-cell motion does not change the tile: Δq and Δoffset cancel.
    nav.update(grid, _portal(-5.3, 2.0), (0.3, 0.0))
    assert nav.player == (1, 0)
    nav.update(grid, _portal(-5.45, 2.0), (0.45, 0.0))
    assert nav.player == (1, 0)
    # Crossing the cell boundary: Δq is one step and the offset wraps by ~1.
    nav.update(grid, _portal(-5.55, 2.0), (-0.45, 0.0))
    assert nav.player == (2, 0)


def test_navmap_plans_through_unknown_cells() -> None:
    """A goal behind a known wall still gets a route (optimistic planning)."""
    layout = load_fixture_layout()
    nav = NavMap(layout)
    for x in range(-2, 3):
        for y in range(-2, 3):
            if x == 0:
                nav.blocked.add((x, y))
            else:
                nav.walkable.add((x, y))
    nav.player = (-2, 0)
    step = nav.step_towards((2, 0))
    assert step is not None
    assert step != (1, 0), "the direct step is a known wall"


def test_navmap_explore_never_stands_still() -> None:
    """With everything reachable already visited, walk to the far end."""
    layout = load_fixture_layout()
    nav = NavMap(layout)
    for x in range(0, 6):
        nav.walkable.add((x, 0))
        nav.visited.add((x, 0))
    nav.player = (0, 0)
    step = nav.explore_step(nav.deepest_cell())
    assert step == (1, 0), "the far end is east"


def test_policy_remembers_a_target_that_left_perception(layout: Layout, make_obs) -> None:
    """Route memory: a goal at the 900 px edge must not be dropped immediately."""
    policy = HeuristicPolicy(layout)
    obs = make_obs().fill_floor().enemy(0, rel_cells=(5.0, 3.0)).player().build()
    first = policy(obs, info_with_offset(), layout)
    assert first[0] != 0
    assert policy._goal is not None
    empty = make_obs().fill_floor().player().build()
    second = policy(empty, info_with_offset(), layout)
    assert policy._goal is not None, "the goal survives a window without perception"
    assert second[0] != 0, "and the body keeps walking to it"


# --------------------------------------------------------------------------- #
# Round 4: pack-fighting micro (issue #94)
# --------------------------------------------------------------------------- #
def test_enemy_timing_table_matches_the_def_resources() -> None:
    """The policy's timing table is the `.tres` data, not a guess."""
    assert _enemy_timing("skeleton_normal") == (0.35, 0.30, 0.40)
    assert _enemy_timing("skeleton_swift") == (0.20, 0.70, 0.35)
    assert _enemy_timing("elite_skeleton") == (0.50, 0.40, 0.50)
    assert _enemy_timing("slime_green") == (0.35, 0.15, 0.40)
    assert _enemy_timing("nonsense") == ENEMY_TIMING["skeleton_normal"], "unknown types default"


def test_time_to_hit_orders_windup_chase_and_recover(layout: Layout, make_obs) -> None:
    """A live telegraph lands soonest; a chase must close first; recover waits."""

    def timings(**enemy_kwargs: Any) -> float:
        builder = make_obs().fill_floor().enemy(0, rel_cells=(0.9, 0.0), **enemy_kwargs)
        builder.set_raw("enemies", "move_speed", 90.0)  # a skeleton_normal
        obs = builder.player(facing=(1.0, 0.0)).build()
        slot = layout.nearest_enemy(obs)
        assert slot is not None
        return _time_to_hit(slot, DEFAULT_CONFIG)

    windup = timings(state="windup", windup_s=0.10)
    chase = timings(state="chase")
    recover = timings(state="recover")
    assert windup == pytest.approx(0.10, abs=1e-3), "a telegraph is its remaining windup"
    assert chase == pytest.approx((128.8 - 50.0) / 90.0 + 0.35, abs=0.05), "close, then wind up"
    assert recover < chase, "a recovery only owes its remaining state + a windup"
    assert windup < recover, "the telegraph is always the most immediate threat"
    assert timings(state="stagger") > 90.0, "a staggered enemy cannot answer"


def test_strafe_keeps_a_perpendicular_step_inside_the_band(layout: Layout, make_obs) -> None:
    """The dodge steps sideways relative to the attacker, not away from it."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.0), state="windup", windup_s=0.2)
        .player(facing=(1.0, 0.0))
        .build()
    )
    slot = layout.nearest_enemy(obs)
    assert slot is not None
    step = _strafe_step(layout, layout.grid(obs), [slot], 0, DEFAULT_CONFIG)
    assert step is not None, "there is always a lateral step in an open room"
    direction = np.asarray(layout.actions[step].screen, dtype=np.float64)
    bearing = np.asarray(slot.screen_delta_cells, dtype=np.float64)
    bearing /= np.linalg.norm(bearing)
    assert abs(float(direction @ bearing)) < 0.75, "the step is lateral, not along the facing"
    # And it does not lunge into point blank.
    after = np.asarray(slot.rel_cells, dtype=np.float64) - np.asarray(
        layout.actions[step].grid, dtype=np.float64
    )
    assert float(np.hypot(*screen_delta(after))) * PX_PER_CELL >= 30.0


def test_strafe_is_deterministic(layout: Layout, make_obs) -> None:
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.2), state="attack", hitbox_active=True)
        .player(facing=(1.0, 0.0))
        .build()
    )
    slot = layout.nearest_enemy(obs)
    assert slot is not None
    first = _strafe_step(layout, layout.grid(obs), [slot], 3, DEFAULT_CONFIG)
    second = _strafe_step(layout, layout.grid(obs), [slot], 3, DEFAULT_CONFIG)
    assert first == second


def test_swing_gate_refuses_a_telegraph_that_lands_first(layout: Layout, make_obs) -> None:
    """64 px is inside the enemy's reach + hurtbox envelope: no trade."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.0), state="windup", windup_s=0.1)
        .player(facing=(1.0, 0.0))
        .build()
    )
    assert heuristic_action(obs, {}, layout)[1] == 0

    # The same distance in a *chase* state loses the race to our swing: it has to
    # close first, so the hit is free and the policy takes it.
    chasing = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    assert heuristic_action(chasing, {}, layout)[1] == 1


def test_pack_gates_are_off_by_default() -> None:
    """The measured-best configuration ships (see the report's tuning log)."""
    assert DEFAULT_CONFIG.strafe_dodge is True
    assert DEFAULT_CONFIG.strafe_side == 1
    assert DEFAULT_CONFIG.strike_margin_px == 60.0
    assert DEFAULT_CONFIG.pack_gate is False
    assert DEFAULT_CONFIG.queue_gate is False
    assert DEFAULT_CONFIG.chokepoint is False
    assert DEFAULT_CONFIG.isolate_target is False
