"""One iteration of a proper DAgger loop for the level-1 warm start (RL-7 r3).

Round 2 cloned the scripted heuristic with two DAgger rounds in a single
process and never measured a round on its own, so it could not tell a clone that
improved from a clone that drifted. This script makes one iteration explicit and
leaves every artifact on disk:

1. **iteration 0** rolls the *demonstrator* (the shipped heuristic) and stores
   the round-0 dataset;
2. **iteration n > 0** rolls the *current student* (``bc_iter{n-1}.zip``), asks
   the demonstrator what it would have done on each visited state, appends the
   relabelled rows to the aggregate on disk (DAgger aggregation), and retrains
   the policy from the previous checkpoint with per-head class-weighted
   cross-entropy;
3. the checkpoint is scored on a **fixed held-out** demonstrator set (per-head
   accuracy, and the attack head's predicted rate against its label rate), and
   — unless ``--no-eval`` — on the closed-loop tuning seeds with the same
   evaluator every other round used.

The closed-loop number is the only one that decides whether DAgger worked; the
held-out numbers are reported because a high accuracy with a collapsed rare
head is exactly what round 2 measured at 0/10.

    uv run python examples/dagger_rounds.py --iteration 0 --work-dir /tmp/rl7/r3
    uv run python examples/dagger_rounds.py --iteration 1 --work-dir /tmp/rl7/r3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from ashcrown_rl.dagger import (  # noqa: E402
    SOURCE_DAGGER,
    SOURCE_DEMO,
    Dataset,
    append_samples,
    head_report,
    load_dataset,
    read_rounds,
    round_row,
    save_dataset,
    write_rounds,
)
from ashcrown_rl.env import AshCrownEnv  # noqa: E402
from ashcrown_rl.evaluate import evaluate_ppo, expand_seed_ranges  # noqa: E402
from ashcrown_rl.history import (  # noqa: E402
    policy_obs_len,
    resolve_frame_stack,
    stack_indices,
    stack_observations,
)
from ashcrown_rl.protocol import ProtocolError  # noqa: E402
from ashcrown_rl.train import (  # noqa: E402
    METADATA_VERSION,
    git_commit,
    write_metadata,
)
from examples.bc_pretrain import (  # noqa: E402
    collect,
    collect_dagger,
    fit,
    head_weights,
    label_stats,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``examples/dagger_rounds.py``."""
    parser = argparse.ArgumentParser(
        prog="python examples/dagger_rounds.py",
        description="One DAgger iteration: collect, aggregate, retrain, evaluate.",
    )
    parser.add_argument("--work-dir", required=True, help="datasets, checkpoints, rounds.csv")
    parser.add_argument("--iteration", type=int, required=True, help="0 = demonstrations")
    parser.add_argument("--demo-steps", type=int, default=64_000, help="round-0 decisions")
    parser.add_argument("--dagger-steps", type=int, default=32_000, help="decisions this round")
    parser.add_argument("--val-steps", type=int, default=16_000, help="held-out decisions")
    parser.add_argument("--val-seed", type=int, default=4242, help="held-out master seed")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7, help="round-0 collection seed")
    parser.add_argument(
        "--dagger-seed-step",
        type=int,
        default=1000,
        help="rollout seed of iteration n is seed + dagger_seed_step * n",
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--character", default="warrior")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument("--epochs", type=int, default=None, help="override the per-round epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--net-arch", default="256,256")
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="observation history length k (docs/07 §5.7; default 4)",
    )
    parser.add_argument("--weight-cap", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dagger-deterministic",
        action="store_true",
        help="roll the student deterministically (default: stochastic, for coverage)",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="iteration 0 starts from this checkpoint instead of a fresh network",
    )
    parser.add_argument(
        "--eval-seeds",
        default="9000-9009",
        help="closed-loop tuning seeds (the 100-seed set is run once, outside this loop)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-timeout", type=float, default=240.0)
    parser.add_argument("--no-eval", action="store_true", help="skip the closed-loop evaluation")
    parser.add_argument("--game-project", default=None)
    parser.add_argument("--godot-bin", default=None)
    return parser


def _dataset_path(work_dir: Path) -> Path:
    return work_dir / "aggregate.npz"


def _val_path(work_dir: Path) -> Path:
    return work_dir / "val_demo.npz"


def _checkpoint_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / f"bc_iter{int(iteration)}.zip"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    clock = time.perf_counter()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    iteration = int(args.iteration)
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
    epochs = int(args.epochs) if args.epochs is not None else (10 if iteration == 0 else 4)
    checkpoint = _checkpoint_path(work_dir, iteration)
    previous = _checkpoint_path(work_dir, iteration - 1)
    try:
        probe = AshCrownEnv(**env_kwargs)
        try:
            classes = [int(value) for value in probe.action_space.nvec]
        finally:
            probe.close()

        # ---- the held-out demonstrator set (fixed across rounds) -----------
        if not _val_path(work_dir).is_file():
            started = time.perf_counter()
            val_obs, val_actions, val_episodes, val_stats = collect(
                n_envs=int(args.n_envs),
                steps=int(args.val_steps),
                seed=int(args.val_seed),
                env_kwargs=env_kwargs,
                frame_stack=stack,
            )
            save_dataset(
                _val_path(work_dir),
                Dataset(
                    obs=val_obs,
                    actions=val_actions,
                    sources=np.full(len(val_obs), SOURCE_DEMO, dtype=np.int8),
                    episode_ids=val_episodes,
                ),
            )
            print(
                "DAGGER_VAL "
                + json.dumps(
                    {"stats": val_stats, "collect_s": round(time.perf_counter() - started, 1)}
                ),
                flush=True,
            )
        val = load_dataset(_val_path(work_dir))
        val_stacked = stack_observations(
            val.obs,
            stack_indices(val.episode_ids, stack) if val.episode_ids is not None else None,
        )

        # ---- collect this round's labels -----------------------------------
        started = time.perf_counter()
        dataset = load_dataset(_dataset_path(work_dir)) if iteration > 0 else None
        if iteration == 0:
            new_obs, new_actions, new_episodes, new_stats = collect(
                n_envs=int(args.n_envs),
                steps=int(args.demo_steps),
                seed=int(args.seed),
                env_kwargs=env_kwargs,
                frame_stack=stack,
            )
            source = SOURCE_DEMO
        else:
            if not previous.is_file():
                raise ValueError(f"missing previous checkpoint {previous}")
            student = PPO.load(str(previous), device=args.device)
            new_obs, new_actions, new_episodes, new_stats = collect_dagger(
                model=student,
                n_envs=int(args.n_envs),
                steps=int(args.dagger_steps),
                seed=int(args.seed) + int(args.dagger_seed_step) * iteration,
                env_kwargs=env_kwargs,
                stochastic=not args.dagger_deterministic,
                frame_stack=stack,
                episode_offset=int(dataset.episode_ids.max()) + 1
                if dataset is not None and dataset.episode_ids is not None
                else 0,
            )
            source = SOURCE_DAGGER
        dataset = append_samples(
            dataset, new_obs, new_actions, source=source, episode_ids=new_episodes
        )
        save_dataset(_dataset_path(work_dir), dataset)
        collect_s = time.perf_counter() - started
        print(
            "DAGGER_DATA "
            + json.dumps(
                {
                    "iteration": iteration,
                    "source": "demo" if source == SOURCE_DEMO else "dagger",
                    "stats": new_stats,
                    "labels": label_stats(new_actions, classes),
                    **dataset.counts(),
                }
            ),
            flush=True,
        )

        # ---- retrain (class-weighted per-head CE, optionally continuing) ---
        started = time.perf_counter()
        weights = head_weights(dataset.actions, classes, args.weight_cap)
        stack_index = (
            stack_indices(dataset.episode_ids, stack)
            if dataset.episode_ids is not None
            else None
        )
        print(
            "DAGGER_STACK "
            + json.dumps(
                {
                    "iteration": iteration,
                    "frame_stack": stack,
                    "stored_obs_shape": list(dataset.obs.shape[1:]),
                    "fit_input_rows": list(stack_index.shape) if stack_index is not None else None,
                    "policy_obs_len": policy_obs_len(int(dataset.obs.shape[1]), stack),
                    "val_stacked_shape": list(val_stacked.shape[1:]),
                }
            ),
            flush=True,
        )
        initial = None
        if iteration == 0 and args.init_checkpoint:
            initial = PPO.load(str(args.init_checkpoint), device=args.device)
        elif iteration > 0:
            initial = PPO.load(str(previous), device=args.device)
        model = fit(
            observations=dataset.obs,
            actions=dataset.actions,
            epochs=epochs,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed) + iteration,
            net_arch=net_arch,
            device=args.device,
            model=initial,
            weights=weights if args.weight_cap > 1.0 else None,
            frame_stack=stack,
            stack_index=stack_index,
        )
        model.save(str(checkpoint))
        fit_s = time.perf_counter() - started
        effective = int(np.prod(model.observation_space.shape))
        print(
            "DAGGER_INPUT "
            + json.dumps(
                {
                    "iteration": iteration,
                    "policy_observation_space": list(model.observation_space.shape),
                    "policy_input_len": effective,
                    "expected": policy_obs_len(int(dataset.obs.shape[1]), stack),
                }
            ),
            flush=True,
        )
        report = head_report(model, val_stacked, val.actions, device=args.device)
        print("DAGGER_ACCURACY " + json.dumps({"iteration": iteration, **report}), flush=True)

        probe = AshCrownEnv(**env_kwargs)
        try:
            write_metadata(
                checkpoint,
                {
                    "version": METADATA_VERSION,
                    "kind": "dagger_bc",
                    "iteration": iteration,
                    "protocol": probe.hello.get("protocol"),
                    "obs_schema": probe.hello.get("obs_schema"),
                    "action_schema": probe.hello.get("action_schema"),
                    "layout_hash": probe.layout_hash,
                    "obs_len": probe.obs_len,
                    "observation_config": probe.observation_config(),
                    "frame_skip": probe.frame_skip,
                    "samples": dataset.samples,
                    "frame_stack": int(stack),
                    "policy_obs_len": policy_obs_len(int(dataset.obs.shape[1]), stack),
                    "demo_samples": dataset.demo_samples,
                    "dagger_samples": dataset.dagger_samples,
                    "epochs": epochs,
                    "weight_cap": float(args.weight_cap),
                    "class_weights": [value.tolist() for value in weights],
                    "demonstrator": "heuristic",
                    "heldout": report,
                    "git": {
                        "training": git_commit(Path(__file__).resolve().parents[1]),
                        "game": git_commit(Path(__file__).resolve().parents[2]),
                    },
                    "command": "uv run python examples/dagger_rounds.py "
                    + " ".join(sys.argv[1:]),
                },
            )
        finally:
            probe.close()

        # ---- closed loop on the tuning seeds -------------------------------
        eval_report = None
        eval_s = 0.0
        if not args.no_eval:
            started = time.perf_counter()
            eval_report = evaluate_ppo(
                checkpoint=str(checkpoint),
                seeds=expand_seed_ranges(args.eval_seeds),
                env_kwargs=env_kwargs,
                level=None,
                workers=int(args.workers),
                csv_path=str(work_dir / f"eval_iter{iteration}.csv"),
                out_path=str(work_dir / f"eval_iter{iteration}.json"),
                seed_timeout=float(args.seed_timeout),
            )
            eval_s = time.perf_counter() - started
            aggregates = eval_report["aggregates"]
            print(
                "DAGGER_EVAL "
                + json.dumps(
                    {
                        "iteration": iteration,
                        "clears": aggregates["clears"],
                        "episodes": aggregates["episodes"],
                        "mean_kills": aggregates["mean_kills"],
                        "mean_steps": aggregates["mean_steps"],
                        "cleared_seeds": [
                            episode["seed"]
                            for episode in eval_report["episodes"]
                            if episode.get("cleared")
                        ],
                        "eval_s": round(eval_s, 1),
                    }
                ),
                flush=True,
            )
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"DAGGER_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    row = round_row(
        iteration=iteration,
        dataset=dataset,
        report=report,
        dagger_stats=new_stats,
        eval_report=eval_report,
        timings={
            "collect_s": collect_s,
            "fit_s": fit_s,
            "eval_s": eval_s,
            "wall_s": time.perf_counter() - clock,
        },
    )
    rounds_path = work_dir / "rounds.csv"
    rows = [existing for existing in read_rounds(rounds_path) if existing["iteration"] != iteration]
    rows.append(row)
    rows.sort(key=lambda entry: int(entry["iteration"]))
    write_rounds(rows, rounds_path)
    print("DAGGER_ROW " + json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - example CLI
    raise SystemExit(main())
