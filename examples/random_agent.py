"""Random-action smoke example for :class:`ashcrown_rl.env.AshCrownEnv` (RL-4).

Samples actions for a few hundred steps, asserts the Gymnasium API contract
(observation shape/dtype/bounds, reward type, terminal flags, ``info`` keys) and
prints the reward components of every step that produced a non-zero reward::

    cd training
    uv run python examples/random_agent.py --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashcrown_rl.env import AshCrownEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the random policy and validate the environment contract."""
    parser = argparse.ArgumentParser(description="Random-action AshCrown smoke run.")
    parser.add_argument("--steps", type=int, default=200, help="decision windows")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--godot-bin", default=None)
    args = parser.parse_args(argv)

    env = AshCrownEnv(
        level=args.level,
        frame_skip=args.frame_skip,
        project=args.game_project,
        godot_bin=args.godot_bin,
    )
    try:
        print(
            f"obs_len={env.obs_len} observation_space={env.observation_space} "
            f"action_space={env.action_space} layout_hash={env.layout_hash[:12]}…"
        )
        obs, info = env.reset(seed=args.seed)
        assert obs.shape == (env.obs_len,), obs.shape
        assert obs.dtype == np.float32, obs.dtype
        assert env.observation_space.contains(obs), "reset observation is outside the space"
        steps = 0
        total = 0.0
        while steps < args.steps and not env.episode_done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += reward
            assert isinstance(reward, float), type(reward)
            assert obs.shape == (env.obs_len,) and obs.dtype == np.float32
            assert np.all(np.isfinite(obs)), "observation holds a non-finite value"
            assert env.observation_space.contains(obs), "step observation is outside the space"
            assert isinstance(info, dict) and "state_hash" in info
            if reward != 0.0:
                components = {
                    key: value
                    for key, value in (info.get("reward_components") or {}).items()
                    if value
                }
                print(f"step {steps:4d} reward={reward:+.3f} components={components}")
            if terminated or truncated:
                print(
                    f"terminal after {steps} steps: reason={info.get('terminal_reason')!r} "
                    f"terminated={terminated} truncated={truncated}"
                )
        print(
            f"ok: {steps} steps, return={total:+.3f}, "
            f"game_seconds={info.get('game_seconds', 0.0):.1f}, "
            f"action_space={env.action_space}"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - example entry point
    raise SystemExit(main())
