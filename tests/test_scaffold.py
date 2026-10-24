"""Bootstrap smoke test: the uv environment and package import work."""

import ashcrown_rl


def test_package_imports() -> None:
    assert ashcrown_rl.__version__ == "0.0.0"
