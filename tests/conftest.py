import pytest
from scverse_misc.logging import config


@pytest.fixture(autouse=True)
def _restore_log_verbosity():
    """Tests that change the shared scverse verbosity must not leak it into other tests."""
    verbosity = config.verbosity
    yield
    config.verbosity = verbosity
