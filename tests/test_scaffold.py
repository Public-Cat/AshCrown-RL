"""Bootstrap smoke tests: the uv environment and the core package imports.

:mod:`ashcrown_rl` is dependency-split: numpy + gymnasium are required, while
``stable-baselines3`` and ``torch`` are the optional ``train`` extra. The
documented gate (``docs/07`` §12.3, ``AGENTS.md``) is a **dev-only** install
(``uv sync --locked --extra dev && uv run pytest``), so importing the package and
its numpy-only modules must never reach the training stack (INT-97: it did, and
collection of three test modules failed with ``ModuleNotFoundError``).

The regression tests below run in a **subprocess** with a meta-path finder that
makes the ``train`` extra invisible, which keeps them meaningful even on a
machine where torch and SB3 are installed: the import graph of the core modules
is checked against the dev-only dependency set, not against the local .venv.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ashcrown_rl

ROOT = Path(__file__).resolve().parents[1]

#: Show a subprocess the dev-only dependency set: ``train`` extra packages are
#: unimportable, exactly as in ``uv sync --locked --extra dev``.
_BLOCK_TRAIN_EXTRAS = '''
import sys


class BlockTrainExtras:
    """Meta-path finder that hides the optional ``train`` extra."""

    blocked = frozenset({"stable_baselines3", "torch", "tensorboard", "pandas"})

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"{fullname} is not installed (blocked by the test)")
        return None


sys.meta_path.insert(0, BlockTrainExtras())
'''

#: Every module the dev-only environment must import without the train extra.
_CORE_MODULES = (
    "ashcrown_rl",
    "ashcrown_rl.baseline",
    "ashcrown_rl.dagger",
    "ashcrown_rl.env",
    "ashcrown_rl.evaluate",
    "ashcrown_rl.history",
    "ashcrown_rl.launch",
    "ashcrown_rl.replay",
    "ashcrown_rl.spaces",
    "ashcrown_rl.train",
    "ashcrown_rl.vec",
)


def _run_without_train_extras(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a subprocess that cannot import the ``train`` extra."""
    code = "\n".join((_BLOCK_TRAIN_EXTRAS, body))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_imports() -> None:
    assert ashcrown_rl.__version__ == "0.0.0"


def test_core_modules_import_with_dev_only_dependencies() -> None:
    imports = "\n".join(f"import {module}" for module in _CORE_MODULES)
    result = _run_without_train_extras(f"{imports}\nprint('core-ok')")
    assert result.returncode == 0, result.stderr
    assert "core-ok" in result.stdout


def test_training_stack_is_imported_lazily() -> None:
    # ``ashcrown_rl.train`` is imported above without SB3; its numpy-only half
    # must work, and every SB3-backed name must stay unresolved until asked for.
    body = """
import ashcrown_rl.train as train

assert train.LevelSchedule([1, 2]).level_for(5) == 2
assert train.CURVE_COLUMNS[0] == "timestep"
for name in ("AnchoredPPO", "Sb3AshCrownVecEnv", "make_training_vec_env", "train_ppo"):
    try:
        getattr(train, name)
    except ImportError:
        continue
    raise SystemExit(f"{name} resolved without stable-baselines3")
print("lazy-ok")
"""
    result = _run_without_train_extras(body)
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout


def test_training_stack_resolves_when_the_extra_is_installed() -> None:
    pytest.importorskip("stable_baselines3", reason="the train extra is not installed")
    pytest.importorskip("torch", reason="the train extra is not installed")
    import ashcrown_rl.train as train

    assert callable(train.train_ppo)
    assert callable(train.make_training_vec_env)
    assert train.Sb3AshCrownVecEnv.__name__ == "Sb3AshCrownVecEnv"
    assert train.AnchoredPPO.__name__ == "AnchoredPPO"
