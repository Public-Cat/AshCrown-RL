"""Behaviour-cloning warm start from the scripted heuristic (RL-7 helper).

PPO cannot bootstrap from a standing start on level 1: an untrained policy
holds the attack pulse so the committed swing (which ignores movement) locks the
player in place for the whole episode, and every episode returns the same
(negative) sum — zero advantage, zero gradient. This script collects heuristic
demonstrations in parallel and fits the PPO policy's action heads by
cross-entropy, so the fine-tuning run starts from a policy that already walks,
faces and swings; ``examples/train_ppo.py --init-checkpoint`` then continues
with PPO (the engine reward, not the demonstrator).

The demonstrator is the *shipped* baseline (``docs/07`` §9.4, 8/10 clears); the
demonstration data is discarded after the fit:

    uv run python examples/bc_pretrain.py --steps 24000 --n-envs 8 \
        --level 1 --out /tmp/rl7/bc_level1.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from ashcrown_rl.baseline import HeuristicPolicy  # noqa: E402
from ashcrown_rl.env import AshCrownEnv  # noqa: E402
from ashcrown_rl.protocol import ProtocolError  # noqa: E402
from ashcrown_rl.train import (  # noqa: E402
    METADATA_VERSION,
    _env_info,
    git_commit,
    make_training_vec_env,
    write_metadata,
)
from ashcrown_rl.vec import make_vec_env  # noqa: E402


def collect(
    *,
    n_envs: int,
    steps: int,
    seed: int,
    env_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the heuristic on ``n_envs`` processes and return ``(obs, actions)``."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    vec = make_vec_env(int(n_envs), seed=int(seed), env_kwargs=env_kwargs)
    try:
        # The layout is published per env; build the policies from a probe env.
        probe = AshCrownEnv(**env_kwargs)
        try:
            policies = [HeuristicPolicy(probe.layout) for _ in range(int(n_envs))]
            probe.close()
        finally:
            probe.close()
        obs, _infos = vec.reset()
        collected = 0
        while collected < int(steps):
            actions_batch = []
            for index in range(int(n_envs)):
                info = _env_info(_infos, index) if isinstance(_infos, dict) else _infos[index]
                actions_batch.append(np.asarray(policies[index](obs[index], info, None)))
            actions_batch_array = np.stack(actions_batch)
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(actions_batch_array.astype(np.int64))
            collected += int(n_envs)
            obs, _rewards, _terminated, _truncated, _infos = vec.step(actions_batch_array)
    finally:
        vec.close()
    return np.concatenate(observations), np.concatenate(actions)


def fit(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    net_arch: list[int],
    device: str = "cpu",
) -> PPO:
    """Fit a fresh PPO policy's action heads to ``actions`` by cross-entropy."""
    env_kwargs = {"level": 1, "frame_skip": 4, "max_ticks": 600}
    vec = make_training_vec_env(n_envs=1, seed=int(seed), env_kwargs=env_kwargs)
    try:
        model = PPO(
            "MlpPolicy",
            vec,
            device=device,
            seed=int(seed),
            verbose=0,
            policy_kwargs={"net_arch": list(net_arch)},
        )
    finally:
        vec.close()
    optimizer = th.optim.Adam(model.policy.parameters(), lr=float(learning_rate))
    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    action_tensor = th.as_tensor(actions, dtype=th.long, device=device)
    count = len(observations)
    rng = np.random.default_rng(int(seed))
    model.policy.set_training_mode(True)
    for epoch in range(int(epochs)):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, int(batch_size)):
            batch = order[start : start + int(batch_size)]
            distribution = model.policy.get_distribution(obs_tensor[batch])
            loss = -distribution.log_prob(action_tensor[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "BC_EPOCH " + json.dumps({"epoch": epoch + 1, "loss": round(total / count, 5)}),
                flush=True,
            )
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``examples/bc_pretrain.py``."""
    parser = argparse.ArgumentParser(
        prog="python examples/bc_pretrain.py",
        description="Behaviour-cloning warm start for PPO from the scripted heuristic.",
    )
    parser.add_argument("--steps", type=int, default=24000, help="demonstration decisions")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel demonstration envs")
    parser.add_argument("--seed", type=int, default=7, help="demonstration master seed")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--character", default="warrior")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--net-arch", default="256,256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--godot-bin", default=None)
    parser.add_argument("--out", required=True, help="checkpoint to write (*.zip)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    env_kwargs: dict[str, Any] = {
        "level": args.level,
        "character": args.character,
        "frame_skip": args.frame_skip,
        "max_ticks": args.max_ticks,
        "godot_bin": args.godot_bin,
        "project": args.game_project,
    }
    net_arch = [int(value) for value in str(args.net_arch).replace(",", " ").split()]
    try:
        observations, actions = collect(
            n_envs=args.n_envs, steps=args.steps, seed=args.seed, env_kwargs=env_kwargs
        )
        print(
            "BC_DATA "
            + json.dumps(
                {"samples": int(len(observations)), "obs_len": int(observations.shape[1])}
            ),
            flush=True,
        )
        model = fit(
            observations=observations,
            actions=actions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            net_arch=net_arch,
            device=args.device,
        )
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"BC_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target))
    probe = AshCrownEnv(**env_kwargs)
    try:
        write_metadata(
            target,
            {
                "version": METADATA_VERSION,
                "kind": "bc_pretrain",
                "protocol": probe.hello.get("protocol"),
                "obs_schema": probe.hello.get("obs_schema"),
                "action_schema": probe.hello.get("action_schema"),
                "layout_hash": probe.layout_hash,
                "obs_len": probe.obs_len,
                "observation_config": probe.observation_config(),
                "frame_skip": probe.frame_skip,
                "samples": int(len(observations)),
                "epochs": int(args.epochs),
                "demonstrator": "heuristic",
                "git": {
                    "training": git_commit(Path(__file__).resolve().parents[1]),
                    "game": git_commit(Path(__file__).resolve().parents[2]),
                },
                "command": "uv run python examples/bc_pretrain.py " + " ".join(sys.argv[1:]),
            },
        )
    finally:
        probe.close()
    print("BC " + json.dumps({"out": str(target), "samples": int(len(observations))}))
    return 0


if __name__ == "__main__":  # pragma: no cover - example CLI
    raise SystemExit(main())
