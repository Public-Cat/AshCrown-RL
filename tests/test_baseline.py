"""Heuristic baseline tests (RL-4, ``docs/07`` §9.4).

The decision rules are unit-tested on synthetic observations built from the
published layout (no process). The integration test is a progress smoke test on
the real level 1: the scripted policy must convert its observation into kills.
The full fixed-seed acceptance run (``>= 8/10`` clear rate) is measured by
``python -m ashcrown_rl.evaluate`` and recorded in
``docs/artifacts/RL/baseline_level1.json`` — see the RL-4 report for the
measured result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ashcrown_rl.baseline import (
    DEFAULT_CONFIG,
    HeuristicConfig,
    HeuristicPolicy,
    frontier_open,
    heuristic_action,
)
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.spaces import PX_PER_CELL, Layout, normalise, screen_delta


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
    # 0.45 grid cells along +x → screen (0.45, 0.225) cells → 0.5 cells →
    # 64 px: inside the warrior's 43+24 px reach, and outside the 60 px
    # point-blank guard, with the enemy chasing.
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.45, 0.0), state="chase")
        .player(facing=(1.0, 0.0))
        .build()
    )
    action = heuristic_action(obs, {}, layout)
    assert action.tolist() == [0, 1, 0]


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


def test_backs_off_during_the_windup(layout: Layout, make_obs) -> None:
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
    assert _dot(action, layout, -enemy.screen_delta_cells) > 0.9, "steps away"


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
    """``min_attack_px`` keeps the policy from swinging at point-blank range."""
    obs = (
        make_obs()
        .fill_floor()
        .enemy(0, rel_cells=(0.25, 0.0), state="chase", cooldown_s=0.0)
        .player(facing=(1.0, 0.0))
        .build()
    )
    default = heuristic_action(obs, {}, layout, DEFAULT_CONFIG)
    assert default.tolist() == [6, 0, 0], "the point-blank guard steps back instead"
    eager = heuristic_action(
        obs, {}, layout, HeuristicConfig(min_attack_px=0.0, safe_cooldown_s=1.0)
    )
    assert eager.tolist() == [0, 1, 0], "with the guard off it swings from 17 px"


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


def test_reach_conversion_matches_the_grid_scale(layout: Layout, make_obs) -> None:
    """A grid step is 128 px per unit in the screen projection (sanity pin)."""
    enemy_screen = screen_delta((1.0, 0.0))
    assert float(np.hypot(*enemy_screen)) * PX_PER_CELL == pytest.approx(
        1.1180339 * PX_PER_CELL, rel=1e-6
    )


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
        .enemy(0, rel_cells=(2.0, 0.0), state="windup", attack_reach_px=50.0)
        .player(facing=(1.0, 0.0))
        .build()
    )
    core = heuristic_action(obs, {}, layout)
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
