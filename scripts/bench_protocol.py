#!/usr/bin/env python3
"""Run the RL-3 protocol benchmark without installing the package.

Equivalent to ``python -m ashcrown_rl.bench`` from the repository root::

    uv run scripts/bench_protocol.py --steps 1000 --csv ../docs/artifacts/RL/bench.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashcrown_rl.bench import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    raise SystemExit(main())
