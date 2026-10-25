"""AshCrown-RL: Python clients and training pipeline for AshCrown.

RL-3 provides the ``ashcrown-rl-protocol-v1`` client (:mod:`ashcrown_rl.protocol`),
the headless launch wrapper (:mod:`ashcrown_rl.launch`) and the step benchmark
(:mod:`ashcrown_rl.bench`). RL-4 adds the Gymnasium environment
(:mod:`ashcrown_rl.env`), the layout/action helpers (:mod:`ashcrown_rl.spaces`),
the vector wrapper (:mod:`ashcrown_rl.vec`), the scripted baseline
(:mod:`ashcrown_rl.baseline`) and the evaluation/replay harnesses
(:mod:`ashcrown_rl.evaluate`, :mod:`ashcrown_rl.replay`). RL-7 is the training
pipeline. See the game repository's ``docs/07-rl-headless-env-spec.md`` for the
contract.
"""

from .baseline import DEFAULT_CONFIG as HEURISTIC_CONFIG
from .baseline import HeuristicConfig, heuristic_action
from .env import AshCrownEnv
from .protocol import (
    AshCrownClient,
    GodotError,
    ProtocolError,
    ProtocolMismatch,
    Timeout,
)
from .spaces import ActionSpec, Layout, LayoutError, load_fixture_layout
from .vec import AshCrownVectorEnv, make_vec_env

__version__ = "0.0.0"

__all__ = [
    "HEURISTIC_CONFIG",
    "ActionSpec",
    "AshCrownClient",
    "AshCrownEnv",
    "AshCrownVectorEnv",
    "GodotError",
    "HeuristicConfig",
    "Layout",
    "LayoutError",
    "ProtocolError",
    "ProtocolMismatch",
    "Timeout",
    "__version__",
    "heuristic_action",
    "load_fixture_layout",
    "make_vec_env",
]
