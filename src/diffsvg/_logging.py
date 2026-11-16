"""diffsvg's logger, the ``scverse.diffsvg`` child of the shared scverse logger.

One setting controls diffsvg and every other package on the shared logger::

    from scverse_misc.logging import config

    config.verbosity = "info"  # default "warning"
"""

from __future__ import annotations

import os

from scverse_misc.logging import get_logger

log = get_logger("diffsvg", timed=True)

#: ``warnings.warn(..., skip_file_prefixes=SKIP_OWN_FRAMES)`` points at the caller's line, not ours.
SKIP_OWN_FRAMES = (os.path.dirname(__file__) + os.sep,)


def report_drop(what: str, dropped: int, total: int, remedy: str = "") -> None:
    """Log that `dropped` of `total` `what` were left out: info, or a warning when half or more are left out."""
    if dropped <= 0:
        return
    emit = log.warning if 2 * dropped >= total else log.info
    emit("left out %d of %d %s", dropped, total, what, deep=remedy or None)
