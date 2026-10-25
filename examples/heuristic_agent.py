"""Heuristic baseline example: clear level 1 with the scripted policy (RL-4).

Runs :func:`ashcrown_rl.baseline.heuristic_action` on a fixed seed, prints one
line per terminal (kill/clear) and the episode summary. This is the scripted
acceptance run of ``docs/07`` §9.4::

    cd training
    uv run python examples/heuristic_agent.py --seed 20260101 --level 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashcrown_rl.baseline import HeuristicPolicy  # noqa: E402
from ashcrown_rl.env import AshCrownEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Play one episode with the heuristic and print what happened."""
    parser = argparse.ArgumentParser(description="Heuristic baseline on one seed.")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--character", default="warrior")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--godot-bin", default=None)
    args = parser.parse_args(argv)

    env = AshCrownEnv(
        level=args.level,
        character=args.character,
        frame_skip=args.frame_skip,
        max_ticks=args.max_ticks,
        project=args.game_project,
        godot_bin=args.godot_bin,
    )
    try:
        obs, info = env.reset(seed=args.seed)
        layout = env.layout
        print(
            f"seed={args.seed} level={args.level} character={args.character} "
            f"obs_len={env.obs_len} layout_hash={env.layout_hash[:12]}…"
        )
        policy = HeuristicPolicy(layout)
        kills = 0
        steps = 0
        total = 0.0
        while not env.episode_done:
            action = policy(obs, info, layout)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += reward
            if int(info.get("kills", 0)) > kills:
                kills = int(info["kills"])
                print(
                    f"  kill #{kills} at {info.get('game_seconds', 0.0):6.1f}s "
                    f"hp={info.get('hp', 0.0):5.1f} enemies_alive={info.get('enemies_alive', 0)}"
                )
            if terminated or truncated:
                break
        print(
            f"terminal={info.get('terminal_reason')!r} steps={steps} "
            f"game_seconds={info.get('game_seconds', 0.0):.1f} "
            f"kills={info.get('kills', 0)} hp={info.get('hp', 0.0):.1f} "
            f"return={total:+.2f} state_hash={info.get('state_hash', '')[:16]}…"
        )
        return 0 if info.get("terminal_reason") == "level_clear" else 1
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover - example entry point
    raise SystemExit(main())
