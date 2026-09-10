"""Project-wide pytest fixtures.

`_isolate_config` is autouse and does two things every test needs.

`Config` caches its instance on the class and pytest never reimports
modules, so a config built in one test would still be there in the next —
hence `Config.reset()` around every case.

And because library code now reaches for `Config()` on its own, a test
that never sets one up would send it through discovery, which starts at
the CWD — under pytest, the repository root, where a real `wsindex.toml`
lives. Pointing `$WSINDEX_CONFIG` at a file that does not exist makes
discovery come up empty (the override wins over every other mode, and a
missing target is not a fallback), so `Config()` yields the built-in
defaults and no test can quietly read the developer's own workspace.
Tests that want real discovery drop the variable themselves — see the
`workspace` fixture in test_cli.py.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from wsindex.config import Config
from wsindex.paths import ENV_OVERRIDE


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "no-such-config.toml"))
    Config.reset()
    yield
    Config.reset()


@pytest.fixture
def anyio_backend() -> str:
    """One event loop implementation for the async tests (MCP's tools).

    asyncio only: trio is not a dependency, and running every async test
    twice would double the suite for no claim it does not already make.
    """
    return "asyncio"
