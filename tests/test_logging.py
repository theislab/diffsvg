import logging
import warnings

import pytest

from diffsvg._logging import report_drop
from diffsvg.tl.model import _fittable_genes


@pytest.mark.parametrize(("dropped", "level"), [(1, logging.INFO), (5, logging.WARNING)])
def test_report_drop_escalates_at_half(caplog, dropped, level):
    with caplog.at_level(logging.DEBUG, logger="scverse.diffsvg"):
        report_drop("genes", dropped, 10)
    (record,) = caplog.records
    assert record.levelno == level
    assert f"left out {dropped} of 10 genes" in record.getMessage()


def test_report_drop_is_silent_without_drops(caplog):
    with caplog.at_level(logging.DEBUG, logger="scverse.diffsvg"):
        report_drop("genes", 0, 10)
    assert not caplog.records


def test_constant_genes_are_reported_not_warned(caplog):
    import numpy as np

    Y = {"ok": np.array([0.0, 1.0, 2.0]), "zero": np.zeros(3), "nan": np.array([0.0, np.nan, 1.0])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.DEBUG, logger="scverse.diffsvg"):
            assert _fittable_genes(list(Y), Y) == ["ok"]
    assert "left out 2 of 3 genes" in caplog.text
