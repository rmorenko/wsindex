"""Project-wide pytest fixtures.

`_reset_config_singleton` is autouse: it clears the `wsindex.config._current`
module slot before and after every test. Without this, any test that calls
`load_config` or `set_config` would leave the current Config visible to the
next test — pytest never reimports modules, so module-level state persists
across the whole session.
"""

from collections.abc import Iterator

import pytest

from wsindex import config as config_module


@pytest.fixture(autouse=True)
def _reset_config_singleton() -> Iterator[None]:
    config_module.reset_config()
    yield
    config_module.reset_config()
