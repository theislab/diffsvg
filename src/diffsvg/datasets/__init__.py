"""Example datasets, downloaded once from the scverse example-data CDN and checked against a pinned sha256."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scverse_misc.datasets import fetch, parse_registry

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["legnini23"]

_BASE_URL, _DATASETS = parse_registry(Path(__file__).parent / "registry.yaml")


def legnini23(cache_dir: str | Path | None = None) -> AnnData:
    """Neural organoid-like cultures with and without SHH induction (Legnini et al. 2023).

    43,762 cells and 88 genes from 17 sections of 8 organoids, 4 with SHH and 4 controls. ``obs`` holds
    ``sample`` (section), ``organoid`` (the biological replicate) and ``condition`` (``"SHH"`` or ``"Ctrl"``);
    ``obsm["spatial"]`` the coordinates; the layers ``raw`` (counts), ``log1p_norm`` and ``norm_ftsqrt``.

    Parameters
    ----------
    cache_dir
        Where to keep the download (about 90 MB). Defaults to the user cache directory, e.g.
        ``~/.cache/diffsvg`` on Linux.

    Returns
    -------
    The dataset as an AnnData object.
    """
    if cache_dir is None:
        import pooch

        cache_dir = pooch.os_cache("diffsvg")
    return fetch(_DATASETS["legnini23"], cache_dir, base_url=_BASE_URL)
