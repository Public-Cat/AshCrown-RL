"""Action-log replay with state-hash verification (RL-4, ``docs/07`` §10.3/§10.4).

A replay log is a small JSON document::

    {"version": 1,
     "pins": {"protocol": 1, "obs_schema": "ashcrown-obs-v2", ...},
     "config": {"level": 1, "character": "warrior", "frame_skip": 4,
                "max_ticks": 18000, "reward_profile": "combat",
                "reward_overrides": {}},
     "seed": 20260101,
     "policy": "heuristic",
     "actions": [[2, 1, 0], ...],
     "hashes": ["…", "…"],
     "states": ["tick=4|level=1|hp=94.20|…", …],
     "rewards": [0.12, …]}

:func:`replay` drives a **fresh** process with the same config/seed/actions and
compares the per-step ``info["state_hash"]`` sequence: a bit-identical replay
proves the determinism contract, and a mismatch names the first differing step
and carries a bounded **state diff** of that step (:func:`state_line`,
:func:`diff_state_lines`) — the report ``docs/07`` §10.4 promises — which makes
the log a usable bug report. ``record`` and ``replay`` are both CLI-friendly::

    uv run python -m ashcrown_rl.replay record --seed 7 --steps 400 --out /tmp/x.json
    uv run python -m ashcrown_rl.replay replay --file /tmp/x.json
"""

from __future__ import annotations

import argparse
import gzip
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

#: Longest state line the replay log keeps per step (bounded failure report).
MAX_STATE_CHARS = 240
#: Longest state diff the mismatch message carries.
MAX_DIFF_CHARS = 320


class ReplayMismatch(AssertionError):
    """The replayed state hashes differ from the recorded ones.

    ``diff`` is the bounded first-difference **state diff** ``docs/07`` §10.4
    promises: the recorded and fresh state lines at the diverging step reduced
    to the fields whose value changed (``name recorded -> fresh``), with the
    last agreeing line kept as context. Logs recorded before ``states``
    existed still raise, they just cannot show the diff.
    """

    def __init__(
        self,
        step: int,
        expected: str,
        actual: str,
        *,
        recorded_state: str = "",
        live_state: str = "",
        previous_state: str = "",
    ) -> None:
        self.step = int(step)
        self.expected = str(expected)
        self.actual = str(actual)
        self.recorded_state = str(recorded_state)[:MAX_STATE_CHARS]
        self.live_state = str(live_state)[:MAX_STATE_CHARS]
        self.previous_state = str(previous_state)[:MAX_STATE_CHARS]
        self.diff = diff_state_lines(self.recorded_state, self.live_state)
        message = (
            f"replay diverged at step {self.step}: recorded hash {self.expected[:16]}… "
            f"but the fresh run produced {self.actual[:16]}…"
        )
        if self.diff:
            message += f"; state diff (recorded -> fresh): {self.diff}"
            if self.previous_state:
                message += f"; last agreeing state: {self.previous_state}"
        elif not self.recorded_state:
            message += "; no state lines in this log (recorded before the §10.4 diff landed)"
        elif not self.live_state:
            message += f"; recorded state {self.recorded_state}; the fresh run gave no state line"
        else:
            message += f"; recorded state {self.recorded_state}; fresh state {self.live_state}"
        super().__init__(message)


# --------------------------------------------------------------------------- #
# State diff (``docs/07`` §10.4: "a divergence fails with the first differing
# tick and a state diff")
# --------------------------------------------------------------------------- #
def _format_value(value: Any, digits: int = 2) -> str:
    """One bounded, JSON-free rendering of a scalar or a short sequence."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(_format_value(item, digits) for item in value)
    return str(value)


def state_line(info: Mapping[str, Any]) -> str:
    """Bounded canonical state line for the replay failure report.

    Summarises the quantities the engine's own canonical ``state_string``
    (``scripts/rl/rl_env.gd``, ``docs/07`` §10.3) covers — tick/level, player
    HP, kills and the living-enemy count, gold, the portal, the boss, the arena
    wave, the absolute player cell/position and the ``stuck`` diagnostic — from
    the ``info`` the client already receives. The engine's string is far too
    large to store per step; this line is capped at :data:`MAX_STATE_CHARS`.
    """

    def get(key: str, default: Any = 0.0) -> Any:
        return info.get(key, default)

    fields = (
        ("tick", _format_value(get("tick", 0), 0)),
        ("level", _format_value(get("level", 0), 0)),
        ("hp", _format_value(get("hp", 0.0))),
        ("kills", _format_value(get("kills", 0), 0)),
        ("alive", _format_value(get("enemies_alive", 0), 0)),
        ("gold", _format_value(get("gold", 0), 0)),
        ("cell", _format_value(get("player_cell", (0.0, 0.0)))),
        ("pos", _format_value(get("player_position", (0.0, 0.0)), 1)),
        (
            "portal",
            f"{_format_value(get('portal_active', False), 0)}"
            f"@{_format_value(get('portal_distance', 0.0))}",
        ),
        ("boss", f"{_format_value(get('boss_hp', 0.0))}/{_format_value(get('boss_phase', 0), 0)}"),
        ("wave", _format_value(get("wave", -1), 0)),
        ("stuck", _format_value(get("stuck", False), 0)),
    )
    return "|".join(f"{name}={value}" for name, value in fields)[:MAX_STATE_CHARS]


def _state_fields(line: str) -> dict[str, str]:
    """``{name: value}`` of one :func:`state_line` string."""
    return {
        name: value
        for name, _, value in (part.partition("=") for part in str(line).split("|"))
        if name
    }


def diff_state_lines(recorded: str, live: str) -> str:
    """Bounded ``name recorded -> fresh`` diff of two state lines.

    Empty when either side is missing (a log without ``states``) or when the
    lines agree field by field. The result never exceeds
    :data:`MAX_DIFF_CHARS`.
    """
    if not recorded or not live:
        return ""
    left = _state_fields(recorded)
    right = _state_fields(live)
    changes = [
        f"{name} {value} -> {right.get(name, '?')}"
        for name, value in left.items()
        if right.get(name, "?") != value
    ]
    text = ", ".join(changes)
    if len(text) > MAX_DIFF_CHARS:
        text = text[: MAX_DIFF_CHARS - 1] + "…"
    return text


def default_policy(
    name: str,
    seed: int,
    layout: Layout,
    *,
    checkpoint: str | None = None,
    env: AshCrownEnv | None = None,
) -> Policy:
    """Policy resolution shared with :mod:`ashcrown_rl.evaluate`.

    ``ppo`` records the actions of a trained checkpoint
    (``--checkpoint model.zip``); the replay itself only needs the actions, so
    a replay log never depends on the policy file.
    """
    if name == "heuristic":
        return HeuristicPolicy(layout)
    if name == "ppo":
        return resolve_policy("ppo", seed, layout, checkpoint=checkpoint, env=env)
    return resolve_policy(name, seed)


def read_log(path: str | Path) -> dict[str, Any]:
    """Read a replay log (``.json`` or gzipped ``.json.gz``)."""
    target = Path(path)
    if target.suffix == ".gz":
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(target.read_text(encoding="utf-8"))


def write_log(path: str | Path, log: Mapping[str, Any]) -> Path:
    """Write a replay log, gzip-compressing when the path ends in ``.gz``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".gz":
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(dict(log), handle, indent=1)
    else:
        target.write_text(json.dumps(dict(log), indent=1) + "\n", encoding="utf-8")
    return target


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
    states: list[str] = []
    rewards: list[float] = []
    terminal: dict[str, Any] = {}
    for _ in range(int(steps)):
        action = policy(obs, info, env.layout)
        actions.append([int(value) for value in action])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        hashes.append(str(info.get("state_hash", "")))
        states.append(state_line(info))
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
            "merchant_policy": env.merchant_policy,
            "hash_every_ticks": env.hash_every_ticks,
            **env.observation_config(),
        },
        "seed": int(seed),
        "policy": policy_name,
        "actions": actions,
        "hashes": hashes,
        "states": states,
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
    states = list(log.get("states") or [])
    if len(hashes) != len(actions):
        raise ValueError(f"log has {len(actions)} actions but {len(hashes)} hashes")
    if len(states) != len(actions):
        states = []  # logs written before the §10.4 state diff landed
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
        merchant_policy=config.get("merchant_policy"),
        hash_every_ticks=int(config.get("hash_every_ticks", 60)),
        obs_grid=bool(config.get("obs_grid", True)),
        obs_grid_radius=int(config.get("obs_grid_radius", 10)),
        perception_px=float(config.get("perception_px", 900.0)),
        expose_portal=bool(config.get("expose_portal", True)),
        obs_blocked=bool(config.get("obs_blocked", False)),
    )
    try:
        if pins.get("layout_hash") and pins["layout_hash"] != target.layout_hash:
            raise ValueError(f"log layout_hash {pins['layout_hash']} != live {target.layout_hash}")
        logged_schema = pins.get("obs_schema")
        if logged_schema and logged_schema != target.layout.obs_schema:
            raise ValueError(
                f"log obs_schema {logged_schema!r} != live {target.layout.obs_schema!r}: "
                "re-record the episode (ashcrown-obs-v2 migration, docs/07 §8.5)"
            )
        obs, info = target.reset(seed=int(log.get("seed", 0)))
        previous_state = state_line(info)
        for step, (action, expected) in enumerate(zip(actions, hashes, strict=True)):
            obs, reward, terminated, truncated, info = target.step(action)
            actual = str(info.get("state_hash", ""))
            live_state = state_line(info)
            if actual != expected:
                raise ReplayMismatch(
                    step,
                    str(expected),
                    actual,
                    recorded_state=str(states[step]) if states else "",
                    live_state=live_state,
                    previous_state=previous_state,
                )
            previous_state = live_state
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
    recorder.add_argument("--policy", default="heuristic", help="heuristic | random | zero | ppo")
    recorder.add_argument(
        "--checkpoint", default=None, help="PPO checkpoint (.zip) recorded by --policy ppo"
    )
    recorder.add_argument("--mode", default="level", help="level | run (RL-5)")
    recorder.add_argument(
        "--merchant-policy", default=None, help="engine merchant sub-policy (none | auto_heal)"
    )
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
                mode=args.mode,
                level=args.level,
                character=args.character,
                frame_skip=args.frame_skip,
                max_ticks=args.max_ticks,
                merchant_policy=args.merchant_policy,
                godot_bin=args.godot_bin,
                project=args.game_project,
            )
            try:
                policy = default_policy(
                    args.policy, args.seed, env.layout, checkpoint=args.checkpoint, env=env
                )
                log = record(
                    env,
                    seed=args.seed,
                    steps=args.steps,
                    policy=policy,
                    policy_name=args.policy,
                )
            finally:
                env.close()
            path = write_log(args.out, log)
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
        log = read_log(args.file)
        config = dict(log.get("config") or {})
        env = AshCrownEnv(
            mode=str(config.get("mode", "level")),
            level=int(config.get("level", 1)),
            character=str(config.get("character", "warrior")),
            frame_skip=int(config.get("frame_skip", 4)),
            max_ticks=int(config.get("max_ticks", 18_000)),
            reward_profile=str(config.get("reward_profile", "combat")),
            reward_overrides=config.get("reward_overrides") or {},
            merchant_policy=config.get("merchant_policy"),
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
    "MAX_DIFF_CHARS",
    "MAX_STATE_CHARS",
    "ReplayMismatch",
    "build_arg_parser",
    "default_policy",
    "diff_state_lines",
    "main",
    "read_log",
    "record",
    "replay",
    "state_line",
    "write_log",
]


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())
