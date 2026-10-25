"""Action-log replay with state-hash verification (RL-4, ``docs/07`` §10.3/§10.4).

A replay log is a small JSON document::

    {"version": 1,
     "pins": {"protocol": 1, "obs_schema": "ashcrown-obs-v1", ...},
     "config": {"level": 1, "character": "warrior", "frame_skip": 4,
                "max_ticks": 18000, "reward_profile": "combat",
                "reward_overrides": {}},
     "seed": 20260101,
     "policy": "heuristic",
     "actions": [[2, 1, 0], ...],
     "hashes": ["…", "…"],
     "rewards": [0.12, …]}

:func:`replay` drives a **fresh** process with the same config/seed/actions and
compares the per-step ``info["state_hash"]`` sequence: a bit-identical replay
proves the determinism contract, and a mismatch names the first differing step
(which makes the log a usable bug report). ``record`` and ``replay`` are both
CLI-friendly::

    uv run python -m ashcrown_rl.replay record --seed 7 --steps 400 --out /tmp/x.json
    uv run python -m ashcrown_rl.replay replay --file /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .baseline import HeuristicPolicy
from .env import AshCrownEnv
from .evaluate import Policy, resolve_policy
from .protocol import ProtocolError
from .spaces import Layout

LOG_VERSION = 1


class ReplayMismatch(AssertionError):
    """The replayed state hashes differ from the recorded ones."""

    def __init__(self, step: int, expected: str, actual: str) -> None:
        self.step = step
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"replay diverged at step {step}: recorded hash {expected[:16]}… "
            f"but the fresh run produced {actual[:16]}…"
        )


def default_policy(name: str, seed: int, layout: Layout) -> Policy:
    """Policy resolution shared with :mod:`ashcrown_rl.evaluate`."""
    if name == "heuristic":
        return HeuristicPolicy(layout)
    return resolve_policy(name, seed)


def record(
    env: AshCrownEnv,
    *,
    seed: int,
    steps: int,
    policy: Policy,
    policy_name: str = "heuristic",
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record ``steps`` actions (or one whole episode) with their state hashes."""
    obs, info = env.reset(seed=int(seed), options=dict(options or {}))
    actions: list[list[int]] = []
    hashes: list[str] = []
    rewards: list[float] = []
    terminal: dict[str, Any] = {}
    for _ in range(int(steps)):
        action = policy(obs, info, env.layout)
        actions.append([int(value) for value in action])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        hashes.append(str(info.get("state_hash", "")))
        if terminated or truncated:
            terminal = {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_reason": str(info.get("terminal_reason", "")),
                "step": len(actions) - 1,
            }
            break
    return {
        "version": LOG_VERSION,
        "pins": {
            "protocol": env.hello.get("protocol"),
            "obs_schema": env.hello.get("obs_schema"),
            "action_schema": env.hello.get("action_schema"),
            "layout_hash": env.layout_hash,
            "obs_len": env.obs_len,
        },
        "config": {
            "level": env.level,
            "character": env.character,
            "mode": env.mode,
            "frame_skip": env.frame_skip,
            "max_ticks": env.max_ticks,
            "reward_profile": env.reward_profile,
            "reward_overrides": dict(env.reward_overrides),
            "hash_every_ticks": env.hash_every_ticks,
        },
        "seed": int(seed),
        "policy": policy_name,
        "actions": actions,
        "hashes": hashes,
        "rewards": rewards,
        "terminal": terminal,
    }


def replay(env: AshCrownEnv | None, log: Mapping[str, Any]) -> dict[str, Any]:
    """Replay a log on ``env`` (or a fresh env built from its config)."""
    if not isinstance(log, Mapping):
        raise ValueError("replay log must be a JSON object")
    version = int(log.get("version", LOG_VERSION))
    if version != LOG_VERSION:
        raise ValueError(f"replay log version {version} is not supported ({LOG_VERSION})")
    hashes = list(log.get("hashes") or [])
    actions = list(log.get("actions") or [])
    if len(hashes) != len(actions):
        raise ValueError(f"log has {len(actions)} actions but {len(hashes)} hashes")
    config = dict(log.get("config") or {})
    pins = dict(log.get("pins") or {})
    built = env is None
    target = env or AshCrownEnv(
        mode=str(config.get("mode", "level")),
        level=int(config.get("level", 1)),
        character=str(config.get("character", "warrior")),
        frame_skip=int(config.get("frame_skip", 4)),
        max_ticks=int(config.get("max_ticks", 18_000)),
        reward_profile=str(config.get("reward_profile", "combat")),
        reward_overrides=config.get("reward_overrides") or {},
        hash_every_ticks=int(config.get("hash_every_ticks", 60)),
    )
    try:
        if pins.get("layout_hash") and pins["layout_hash"] != target.layout_hash:
            raise ValueError(f"log layout_hash {pins['layout_hash']} != live {target.layout_hash}")
        obs, info = target.reset(seed=int(log.get("seed", 0)))
        for step, (action, expected) in enumerate(zip(actions, hashes, strict=True)):
            obs, reward, terminated, truncated, info = target.step(action)
            actual = str(info.get("state_hash", ""))
            if actual != expected:
                raise ReplayMismatch(step, str(expected), actual)
            del reward
            if terminated or truncated:
                if step != len(actions) - 1:
                    raise ValueError(
                        f"episode ended at step {step} but the log continues to {len(actions) - 1}"
                    )
                break
        return {
            "ok": True,
            "steps": len(actions),
            "seed": int(log.get("seed", 0)),
            "terminal_reason": str(info.get("terminal_reason", "")),
            "layout_hash": target.layout_hash,
        }
    finally:
        if built:
            target.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m ashcrown_rl.replay``."""
    parser = argparse.ArgumentParser(
        prog="python -m ashcrown_rl.replay",
        description="Record an action log and replay it against a fresh process.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    recorder = subparsers.add_parser("record", help="record a short episode")
    recorder.add_argument("--seed", type=int, default=20260101)
    recorder.add_argument("--steps", type=int, default=400)
    recorder.add_argument("--level", type=int, default=1)
    recorder.add_argument("--character", default="warrior")
    recorder.add_argument("--frame-skip", type=int, default=4)
    recorder.add_argument("--max-ticks", type=int, default=18_000)
    recorder.add_argument("--policy", default="heuristic", help="heuristic | random | zero")
    recorder.add_argument("--game-project", default=None)
    recorder.add_argument("--godot-bin", default=None)
    recorder.add_argument("--out", required=True, help="replay log to write")
    player = subparsers.add_parser("replay", help="replay a log and verify the hashes")
    player.add_argument("--file", required=True, help="replay log to read")
    player.add_argument("--game-project", default=None)
    player.add_argument("--godot-bin", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "record":
            env = AshCrownEnv(
                level=args.level,
                character=args.character,
                frame_skip=args.frame_skip,
                max_ticks=args.max_ticks,
                godot_bin=args.godot_bin,
                project=args.game_project,
            )
            try:
                policy = default_policy(args.policy, args.seed, env.layout)
                log = record(
                    env,
                    seed=args.seed,
                    steps=args.steps,
                    policy=policy,
                    policy_name=args.policy,
                )
            finally:
                env.close()
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
            print(
                "REPLAY_RECORD "
                + json.dumps(
                    {
                        "out": str(path),
                        "seed": log["seed"],
                        "steps": len(log["actions"]),
                        "policy": log["policy"],
                        "terminal": log["terminal"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        log = json.loads(Path(args.file).read_text(encoding="utf-8"))
        config = dict(log.get("config") or {})
        env = AshCrownEnv(
            mode=str(config.get("mode", "level")),
            level=int(config.get("level", 1)),
            character=str(config.get("character", "warrior")),
            frame_skip=int(config.get("frame_skip", 4)),
            max_ticks=int(config.get("max_ticks", 18_000)),
            reward_profile=str(config.get("reward_profile", "combat")),
            reward_overrides=config.get("reward_overrides") or {},
            hash_every_ticks=int(config.get("hash_every_ticks", 60)),
            godot_bin=args.godot_bin,
            project=args.game_project,
        )
        try:
            result = replay(env, log)
        finally:
            env.close()
        print("REPLAY " + json.dumps(result, sort_keys=True))
        return 0
    except (ProtocolError, OSError, ValueError, ReplayMismatch) as exc:
        print(f"REPLAY_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "LOG_VERSION",
    "ReplayMismatch",
    "build_arg_parser",
    "default_policy",
    "main",
    "record",
    "replay",
]
