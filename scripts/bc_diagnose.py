"""Measure a behaviour-cloned checkpoint against fresh heuristic demonstrations.

RL-7 round 2 helper. The question this answers, with numbers instead of
hypothesis: *does the cloned policy actually reproduce the demonstrator on data
it has never seen*, and if not, which of its three action heads is to blame?

It reports, per head (``move``/``attack``/``interact``):

* argmax accuracy on held-out demonstration states,
* the label distribution versus the predicted distribution (this is how a
  collapsed "never attack" head shows up),
* the mean predicted probability of the chosen class (calibration),
* the *mean log-probability* of the expert action under the policy.

``--check-train-seed`` additionally measures the same metrics on demonstrations
drawn from the seed ``examples/bc_pretrain.py`` was trained on, so a
generalisation gap is distinguishable from a fit gap. ``--verify-config``
compares the checkpoint sidecar's observation pins against a live environment
without collecting anything.

    uv run python scripts/bc_diagnose.py --checkpoint /tmp/rl7/bc2/bc_level1.zip \
        --steps 12800 --n-envs 16 --out /tmp/rl7/diag/bc2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import numpy as np  # noqa: E402
from bc_pretrain import collect  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from ashcrown_rl.env import AshCrownEnv  # noqa: E402
from ashcrown_rl.train import read_metadata  # noqa: E402

#: Human-readable names of the three ``ashcrown-act-v1`` heads.
HEADS: tuple[str, ...] = ("move", "attack", "interact")


def head_metrics(
    name: str, labels: np.ndarray, logits: np.ndarray, *, confusion: bool = False
) -> dict[str, Any]:
    """Accuracy and distribution statistics for one categorical head."""
    predicted = np.argmax(logits, axis=1)
    classes = int(logits.shape[1])
    label_counts = [int(np.sum(labels == index)) for index in range(classes)]
    predicted_counts = [int(np.sum(predicted == index)) for index in range(classes)]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    chosen = probabilities[np.arange(len(labels)), labels]
    payload: dict[str, Any] = {
        "head": name,
        "classes": classes,
        "accuracy": float(np.mean(predicted == labels)),
        "labels": label_counts,
        "predicted": predicted_counts,
        "label_rate": [round(count / max(len(labels), 1), 5) for count in label_counts],
        "predicted_rate": [
            round(count / max(len(labels), 1), 5) for count in predicted_counts
        ],
        "mean_prob_of_label": float(np.mean(chosen)),
        "mean_nll": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
    }
    if confusion:
        matrix = np.zeros((classes, classes), dtype=np.int64)
        for label, guess in zip(labels.tolist(), predicted.tolist(), strict=True):
            matrix[int(label), int(guess)] += 1
        payload["confusion"] = matrix.tolist()
    return payload


def measure(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    confusion: bool = False,
) -> dict[str, Any]:
    """Run the cloned policy over ``(obs, action)`` pairs and summarise."""
    import torch as th  # noqa: PLC0415 - only the diagnostic needs torch typing

    with th.no_grad():
        tensor = th.as_tensor(observations, dtype=th.float32)
        distribution = model.policy.get_distribution(tensor)
        # ``MultiCategoricalDistribution.distribution`` is the list of the
        # three per-head ``Categorical`` objects (SB3 >= 2.x).
        logits = [head.logits for head in distribution.distribution]
        per_head = [
            head_metrics(
                HEADS[index], actions[:, index], np.asarray(logit), confusion=confusion
            )
            for index, logit in enumerate(logits)
        ]
        joint = np.stack([np.argmax(np.asarray(logit), axis=1) for logit in logits], axis=1)
    return {
        "samples": int(len(observations)),
        "exact_match": float(np.mean(np.all(joint == actions, axis=1))),
        "heads": per_head,
    }


def verify_config(checkpoint: Path, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Compare the checkpoint sidecar's observation pins with a live handshake."""
    metadata = read_metadata(checkpoint)
    env = AshCrownEnv(**env_kwargs)
    try:
        live = {
            "protocol": env.hello.get("protocol"),
            "obs_schema": env.hello.get("obs_schema"),
            "layout_hash": env.layout_hash,
            "obs_len": env.obs_len,
            "observation_config": env.observation_config(),
        }
    finally:
        env.close()
    pinned = {key: metadata.get(key) for key in live}
    return {
        "identical": pinned == live,
        "pinned": pinned,
        "live": live,
        "differences": [key for key in live if pinned.get(key) != live.get(key)],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``scripts/bc_diagnose.py``."""
    parser = argparse.ArgumentParser(prog="python scripts/bc_diagnose.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=12800, help="held-out decisions")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242, help="held-out demo master seed")
    parser.add_argument(
        "--check-train-seed",
        type=int,
        default=None,
        help="also measure demonstrations from this (training) master seed",
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--character", default="warrior")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=18_000)
    parser.add_argument(
        "--dataset",
        default=None,
        help="measure stored rows (the DAgger aggregate) instead of fresh demos",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="with --dataset: use only the last N rows (one DAgger round's own rows)",
    )
    parser.add_argument(
        "--source",
        choices=("all", "demo", "dagger"),
        default="all",
        help="with --dataset: keep only rows of this provenance",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="also report each head's label-vs-predicted confusion matrix",
    )
    parser.add_argument("--verify-config", action="store_true")
    parser.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    env_kwargs: dict[str, Any] = {
        "level": args.level,
        "character": args.character,
        "frame_skip": args.frame_skip,
        "max_ticks": args.max_ticks,
    }
    checkpoint = Path(args.checkpoint)
    report: dict[str, Any] = {"checkpoint": str(checkpoint)}
    if args.verify_config:
        report["config"] = verify_config(checkpoint, env_kwargs)
        print("BC_CONFIG " + json.dumps(report["config"]), flush=True)
    model = PPO.load(str(checkpoint), device="cpu")
    if args.dataset:
        from ashcrown_rl.dagger import SOURCE_DAGGER, SOURCE_DEMO, load_dataset  # noqa: PLC0415

        dataset = load_dataset(args.dataset)
        keep = np.ones(dataset.samples, dtype=bool)
        if args.source != "all":
            wanted = SOURCE_DAGGER if args.source == "dagger" else SOURCE_DEMO
            keep &= dataset.sources == wanted
        if int(args.tail) > 0:
            rows = np.flatnonzero(keep)[-int(args.tail) :]
            keep = np.zeros(dataset.samples, dtype=bool)
            keep[rows] = True
        report["dataset"] = {
            "path": str(args.dataset),
            "source": args.source,
            "tail": int(args.tail),
            **measure(
                model,
                dataset.obs[keep],
                dataset.actions[keep],
                confusion=True,
            ),
        }
        print("BC_DATASET " + json.dumps(report["dataset"]), flush=True)
        if args.out:
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    for label, seed in (("held_out", args.seed), ("train", args.check_train_seed)):
        if seed is None:
            continue
        observations, actions = collect(
            n_envs=args.n_envs, steps=args.steps, seed=int(seed), env_kwargs=env_kwargs
        )
        report[label] = {"seed": int(seed), **measure(model, observations, actions)}
        print(f"BC_{label.upper()} " + json.dumps(report[label]), flush=True)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("BC_OUT " + str(target), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - operator script
    raise SystemExit(main())
