"""Behaviour-cloning warm start from the scripted heuristic (RL-7 helper).

PPO cannot bootstrap from a standing start on level 1: an untrained policy holds
the attack pulse so the committed swing (which ignores movement) locks the player
in place for the whole episode, and every episode returns the same (negative)
sum — zero advantage, zero gradient. This script collects heuristic
demonstrations and fits the PPO policy's action heads by cross-entropy, so the
fine-tuning run starts from a policy that already walks, faces and swings;
``examples/train_ppo.py --init-checkpoint`` then continues with PPO (the engine
reward, not the demonstrator).

The demonstrator is the *shipped* baseline (``docs/07`` §9.4, 8/10 clears).

Two things make the clone usable rather than merely well-fitting
(``RL7-report.md`` §5, round 2):

* **per-head class weighting** — the attack head fires on ~2-3 % of the windows
  and the interact head on ~0.1 %, so a plain cross-entropy collapses both to
  "never"; the weights are inverse class frequency (capped by ``--weight-cap``),
  which restores the demonstrator's attack rate;
* **DAgger rounds** — a clone that only ever sees demonstrator states suffers
  covariate shift: a 72 % per-window move accuracy is enough to leave the
  demonstrator's manifold within a second, and off-manifold there are no labels
  at all. Each round rolls the *current policy*, asks the heuristic what it
  would have done on those states, and retrains on the aggregate.

    uv run python examples/bc_pretrain.py --steps 144000 --n-envs 16 --level 1 \
        --epochs 12 --dagger-rounds 2 --dagger-steps 48000 \
        --out /tmp/rl7/bc3/bc_level1.zip
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
from ashcrown_rl.history import (  # noqa: E402
    policy_obs_len,
    resolve_frame_stack,
    stack_indices,
)
from ashcrown_rl.protocol import ProtocolError  # noqa: E402
from ashcrown_rl.train import (  # noqa: E402
    METADATA_VERSION,
    _env_info,
    git_commit,
    make_training_vec_env,
    write_metadata,
)
from ashcrown_rl.vec import make_vec_env  # noqa: E402


def _label_batch(
    policies: list[HeuristicPolicy],
    obs: np.ndarray,
    infos: Any,
    n_envs: int,
) -> np.ndarray:
    """The heuristic's action for every env of one vector step."""
    actions = []
    for index in range(n_envs):
        info = _env_info(infos, index) if isinstance(infos, dict) else infos[index]
        actions.append(np.asarray(policies[index](obs[index], info, None)))
    return np.stack(actions).astype(np.int64)


def collect(
    *,
    n_envs: int,
    steps: int,
    seed: int,
    env_kwargs: dict[str, Any],
    frame_stack: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Roll the heuristic on ``n_envs`` processes and label its own states.

    Returns ``(raw_obs, actions, episode_ids, stats)``. Each row is
    ``(observation at decision t, action the heuristic chose at that same
    observation)`` — the pairing is built in one statement per step, before the
    environment is advanced, so no off-by-one is possible.

    With ``frame_stack > 1`` the *worker env* publishes the stacked window; only
    the newest frame (``obs[:, -1]``) is stored, together with the collection
    episode id, and
    :func:`~ashcrown_rl.history.stack_indices` rebuilds the identical window at
    fit time (see :mod:`ashcrown_rl.dagger`).
    """
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    stack = resolve_frame_stack(frame_stack)
    counters = np.zeros(int(n_envs), dtype=np.int64)
    vec = make_vec_env(int(n_envs), seed=int(seed), env_kwargs=env_kwargs, frame_stack=stack)
    stats = {"episodes": 0, "clears": 0, "frame_stack": stack}
    try:
        probe = AshCrownEnv(**env_kwargs)
        try:
            policies = [HeuristicPolicy(probe.layout) for _ in range(int(n_envs))]
        finally:
            probe.close()
        obs, infos = vec.reset()
        collected = 0
        while collected < int(steps):
            actions_batch = _label_batch(policies, obs, infos, int(n_envs))
            observations.append(_raw_frame(obs))
            actions.append(actions_batch)
            episodes.append(counters.copy())
            collected += int(n_envs)
            obs, _rewards, terminated, _truncated, infos = vec.step(actions_batch)
            stats["episodes"] += int(np.sum(terminated))
            stats["clears"] += _clear_count(infos, int(n_envs))
            for index in range(int(n_envs)):
                if int(_env_info(infos, index).get("episode_steps", 1) or 0) == 0:
                    counters[index] += 1
                    # Auto-reset: the demonstrator's memory (nav map, pocket
                    # guard, sweep ring) belongs to the *previous* level. Leaking
                    # it across episodes measurably breaks the demonstrator —
                    # the RL-7 round-2 measurement was 0 clears in 25 episodes
                    # with a leaked policy versus 8/10 with a fresh one.
                    policies[index] = HeuristicPolicy(policies[index].layout)
    finally:
        vec.close()
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        np.concatenate(episodes),
        stats,
    )


def _raw_frame(obs: Any) -> np.ndarray:
    """The newest raw frame of a (possibly stacked) batched observation."""
    values = np.asarray(obs, dtype=np.float32)
    if values.ndim == 3:  # (n_envs, frame_stack, obs_len)
        return values[:, -1, :].copy()
    return values.copy()


def _clear_count(infos: Any, n_envs: int) -> int:
    """How many envs of this vector step finished on the level-clear terminal.

    With ``autoreset_mode=SAME_STEP`` the *top-level* info of a finished env is
    already the **reset** info of the next episode; the finished episode's own
    info is nested under ``final_info``. Reading ``terminal_reason`` from the top
    level reports 0 clears even when every episode cleared (the RL-7 round-2
    first measurement bug).
    """
    clears = 0
    for index in range(n_envs):
        info = _env_info(infos, index)
        final = info.get("final_info") or {}
        reason = str(info.get("terminal_reason") or final.get("terminal_reason") or "")
        if reason in ("level_clear", "portal_entered"):
            clears += 1
    return clears


def collect_dagger(
    *,
    model: PPO,
    n_envs: int,
    steps: int,
    seed: int,
    env_kwargs: dict[str, Any],
    stochastic: bool = True,
    frame_stack: int = 1,
    episode_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Roll ``model`` and label its own visited states with the heuristic (DAgger)."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    stack = resolve_frame_stack(frame_stack)
    counters = np.zeros(int(n_envs), dtype=np.int64) + int(episode_offset)
    vec = make_vec_env(int(n_envs), seed=int(seed), env_kwargs=env_kwargs, frame_stack=stack)
    stats = {"episodes": 0, "clears": 0, "frame_stack": stack}
    try:
        probe = AshCrownEnv(**env_kwargs)
        try:
            policies = [HeuristicPolicy(probe.layout) for _ in range(int(n_envs))]
        finally:
            probe.close()
        obs, infos = vec.reset()
        collected = 0
        while collected < int(steps):
            labels = _label_batch(policies, obs, infos, int(n_envs))
            # The executed action comes from the *policy*, the label from the
            # demonstrator; only the label is learned.
            executed = np.stack(
                [
                    np.asarray(model.predict(obs[index], deterministic=not stochastic)[0])
                    for index in range(int(n_envs))
                ]
            ).astype(np.int64)
            observations.append(_raw_frame(obs))
            actions.append(labels)
            episodes.append(counters.copy())
            collected += int(n_envs)
            obs, _rewards, terminated, _truncated, infos = vec.step(executed)
            stats["episodes"] += int(np.sum(terminated))
            stats["clears"] += _clear_count(infos, int(n_envs))
            for index in range(int(n_envs)):
                info = _env_info(infos, index)
                if int(info.get("episode_steps", 1) or 0) == 0:
                    # Auto-reset happened for this env: the heuristic's memory
                    # (nav map, pocket guard, sweep) belongs to the old episode.
                    policies[index] = HeuristicPolicy(policies[index].layout)
                    counters[index] += 1
    finally:
        vec.close()
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        np.concatenate(episodes),
        stats,
    )


def head_weights(
    actions: np.ndarray, classes: list[int], cap: float
) -> list[np.ndarray]:
    """Inverse-frequency class weights, capped per class (``cap`` disables at 1)."""
    weights: list[np.ndarray] = []
    for index, count in enumerate(classes):
        labels = actions[:, index]
        totals = np.bincount(labels, minlength=int(count)).astype(np.float64)
        present = np.maximum(totals, 1.0)
        weight = totals.sum() / (float(count) * present)
        weights.append(np.clip(weight, 1.0, max(float(cap), 1.0)).astype(np.float32))
    return weights


def label_stats(actions: np.ndarray, classes: list[int]) -> list[dict[str, Any]]:
    """Per-head label counts, for the log (an all-zero head is a dead head)."""
    return [
        {
            "head": index,
            "counts": [
                int(value)
                for value in np.bincount(actions[:, index], minlength=int(count))
            ],
        }
        for index, count in enumerate(classes)
    ]


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
    model: PPO | None = None,
    weights: list[np.ndarray] | None = None,
    frame_stack: int = 1,
    stack_index: np.ndarray | None = None,
) -> PPO:
    """Fit the policy's action heads to ``actions`` by (weighted) cross-entropy.

    ``stack_index`` (from :func:`~ashcrown_rl.history.stack_indices`) turns the
    stored **raw** observations into the ``(k, obs_len)`` windows the policy was
    trained on; without it the rows are fed single-frame.
    """
    stack = resolve_frame_stack(frame_stack)
    if model is None:
        env_kwargs = {"level": 1, "frame_skip": 4, "max_ticks": 600}
        vec = make_training_vec_env(
            n_envs=1, seed=int(seed), env_kwargs=env_kwargs, frame_stack=stack
        )
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
    weight_tensors = (
        [th.as_tensor(value, dtype=th.float32, device=device) for value in weights]
        if weights is not None
        else None
    )
    for epoch in range(int(epochs)):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, int(batch_size)):
            batch = order[start : start + int(batch_size)]
            if stack_index is not None:
                window = stack_index[batch].reshape(-1)
            else:
                window = batch
            heads = model.policy.get_distribution(obs_tensor[window]).distribution
            losses = []
            for index, head in enumerate(heads):
                log_prob = head.log_prob(action_tensor[batch, index])
                if weight_tensors is not None:
                    log_prob = log_prob * weight_tensors[index][action_tensor[batch, index]]
                losses.append(-log_prob.mean())
            loss = losses[0]
            for extra in losses[1:]:
                loss = loss + extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0:
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
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="observation history length k (docs/07 §5.7; default 4)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=1.0,
        help="cap on the inverse-frequency class weight (1 = plain cross-entropy)",
    )
    parser.add_argument("--dagger-rounds", type=int, default=0, help="DAgger iterations")
    parser.add_argument("--dagger-steps", type=int, default=32_000, help="decisions per round")
    parser.add_argument(
        "--dagger-deterministic",
        action="store_true",
        help="roll the DAgger policy deterministically (default: stochastic, for coverage)",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="continue cloning an existing checkpoint instead of a fresh net",
    )
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
    stack = resolve_frame_stack(args.frame_stack)
    try:
        observations, actions, episode_ids, collect_stats = collect(
            n_envs=args.n_envs,
            steps=args.steps,
            seed=args.seed,
            env_kwargs=env_kwargs,
            frame_stack=stack,
        )
        stack_index = stack_indices(episode_ids, stack)
        probe = AshCrownEnv(**env_kwargs)
        try:
            classes = [int(value) for value in probe.action_space.nvec]
        finally:
            probe.close()
        print(
            "BC_DATA "
            + json.dumps(
                {
                    "samples": int(len(observations)),
                    "obs_len": int(observations.shape[1]),
                    "frame_stack": stack,
                    "policy_obs_len": policy_obs_len(int(observations.shape[1]), stack),
                    "stats": collect_stats,
                    "labels": label_stats(actions, classes),
                }
            ),
            flush=True,
        )
        weights = head_weights(actions, classes, args.weight_cap)
        if args.weight_cap > 1.0:
            print(
                "BC_WEIGHTS "
                + json.dumps({"weights": [value.tolist() for value in weights]}),
                flush=True,
            )
        initial = (
            PPO.load(str(args.init_checkpoint), device=args.device)
            if args.init_checkpoint
            else None
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
            model=initial,
            weights=weights if args.weight_cap > 1.0 else None,
            frame_stack=stack,
            stack_index=stack_index,
        )
        for round_index in range(int(args.dagger_rounds)):
            new_obs, new_actions, new_episodes, stats = collect_dagger(
                model=model,
                n_envs=args.n_envs,
                steps=args.dagger_steps,
                seed=int(args.seed) + 1000 + round_index,
                env_kwargs=env_kwargs,
                stochastic=not args.dagger_deterministic,
                frame_stack=stack,
                episode_offset=int(episode_ids.max()) + 1 if episode_ids.size else 0,
            )
            observations = np.concatenate([observations, new_obs])
            actions = np.concatenate([actions, new_actions])
            episode_ids = np.concatenate([episode_ids, new_episodes])
            stack_index = stack_indices(episode_ids, stack)
            print(
                "BC_DAGGER "
                + json.dumps(
                    {
                        "round": round_index + 1,
                        "samples": int(len(observations)),
                        "new": int(len(new_obs)),
                        "stats": stats,
                        "labels": label_stats(new_actions, classes),
                    }
                ),
                flush=True,
            )
            weights = head_weights(actions, classes, args.weight_cap)
            model = fit(
                observations=observations,
                actions=actions,
                epochs=max(3, int(args.epochs) // 2),
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=int(args.seed) + 1 + round_index,
                net_arch=net_arch,
                device=args.device,
                model=model,
                weights=weights if args.weight_cap > 1.0 else None,
                frame_stack=stack,
                stack_index=stack_index,
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
                "frame_stack": int(stack),
                "policy_obs_len": policy_obs_len(int(observations.shape[1]), stack),
                "epochs": int(args.epochs),
                "weight_cap": float(args.weight_cap),
                "dagger_rounds": int(args.dagger_rounds),
                "dagger_steps": int(args.dagger_steps),
                "class_weights": [value.tolist() for value in weights],
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
