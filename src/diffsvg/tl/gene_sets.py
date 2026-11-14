"""Gene-set tests on per-slide summaries of the spatial curves."""

import warnings

import numpy as np
import pandas as pd

from diffsvg.tl.model import _anchor_support, _benjamini_hochberg


def replicate_summaries(model_setup):
    """Level and slope of every gene in every slide, on the anchor range where the cells are.

    Per slide and gene, a line is fitted to the binned means by weighted least squares (weights: cells per
    bin) over the anchor support (see `fit_model`). The level is the line's value at the middle of the
    support, the slope its slope in units of the full ``[0, 1]`` anchor range.

    Parameters
    ----------
    model_setup
        `setup_model`'s output for data binned with `sample` (`diffsvg.pp.bin_obs`).

    Returns
    -------
    level, slope : pandas.DataFrame
        Slides x genes.
    condition : pandas.Series
        The condition (0 or 1, as encoded by `diffsvg.pp.bin_obs`) of every slide.
    """
    binned = model_setup.binned
    if binned.sample is None or binned.n_obs is None:
        raise ValueError("Gene-set tests need data binned with `sample` and contiguous bins (`window_radius=None`).")
    lo, hi = _anchor_support(binned)
    B, B_pred = np.asarray(binned.B), np.asarray(binned.B_pred)
    # every binned row sits on a grid point; find it from its basis row
    x = np.asarray(binned.bin_centers)[((B[:, None, :] - B_pred[None, :, :]) ** 2).sum(-1).argmin(1)]
    w = np.asarray(binned.n_obs, dtype=float)
    slide = np.asarray(binned.sample_index)
    cond = np.asarray(binned.condition_fit)
    Y = np.column_stack([np.asarray(binned.Y[g], dtype=float) for g in binned.genes])
    labels = list(binned.sample_labels)
    level = np.full((len(labels), len(binned.genes)), np.nan)
    slope = np.full_like(level, np.nan)
    condition = np.full(len(labels), -1)
    for s in range(len(labels)):
        rows = (slide == s) & (x >= lo) & (x <= hi) & (w > 0)
        if rows.sum() < 2:
            continue
        X = np.column_stack([np.ones(rows.sum()), x[rows] - (lo + hi) / 2])
        beta = np.linalg.lstsq(X * np.sqrt(w[rows, None]), Y[rows] * np.sqrt(w[rows, None]), rcond=None)[0]
        level[s], slope[s], condition[s] = beta[0], beta[1], cond[rows][0]
    keep = condition >= 0
    index = pd.Index(np.asarray(labels)[keep], name=binned.sample)
    return (
        pd.DataFrame(level[keep], index=index, columns=binned.genes),
        pd.DataFrame(slope[keep], index=index, columns=binned.genes),
        pd.Series(condition[keep], index=index, name="condition"),
    )


def _as_sets(gene_sets, genes):
    """``{name: (gene indices, weights)}`` from a dict of gene lists or a decoupler-style ``source/target[/weight]`` frame."""
    if isinstance(gene_sets, pd.DataFrame):
        frame = gene_sets if "weight" in gene_sets else gene_sets.assign(weight=1.0)
        items = {name: dict(zip(g["target"], g["weight"], strict=True)) for name, g in frame.groupby("source")}
    else:
        items = {name: dict.fromkeys(members, 1.0) for name, members in gene_sets.items()}
    position = {g: i for i, g in enumerate(genes)}
    out = {}
    for name, members in items.items():
        hit = [(position[g], float(v)) for g, v in members.items() if g in position]
        if hit:
            idx, wts = zip(*hit, strict=True)
            out[name] = (np.array(idx), np.array(wts))
    return out


def _t_from_rotated(U):
    """Row 0 of `U` is the contrast, the other rows the residual directions: per-gene t."""
    s = np.sqrt((U[1:] ** 2).mean(axis=0))
    return U[0] / np.where(s > 0, s, np.inf)


def _rotation_pvalues(Ymat, group, sets, n_rot, rng):
    """Self-contained rotation test (Wu et al. 2010, limma's roast) with the slide as the unit."""
    n = len(group)
    Q = np.linalg.qr(np.column_stack([np.ones(n), group]), mode="complete")[0]
    U = Q.T @ Ymat
    sign = np.sign(Q[:, 1] @ (group - group.mean()))  # make a positive contrast mean "higher in condition 1"
    U = U[1:] * np.r_[sign, np.ones(n - 2)][:, None]
    t_obs = _t_from_rotated(U)

    def stats(t):
        return {k: ((w * t[i]).sum() / np.abs(w).sum(), (np.abs(w) * np.abs(t[i])).sum() / np.abs(w).sum()) for k, (i, w) in sets.items()}

    observed = stats(t_obs)
    counts = {k: np.zeros(3) for k in sets}
    for _ in range(n_rot):
        R, upper = np.linalg.qr(rng.standard_normal((n - 1, n - 1)))
        R = R * np.sign(np.diag(upper))  # Haar-uniform; a plain QR is not
        for k, (m, a) in stats(_t_from_rotated(R @ U)).items():
            counts[k] += [m >= observed[k][0], m <= observed[k][0], a >= observed[k][1]]
    return {k: (observed[k][0], *((counts[k] + 1) / (n_rot + 1))) for k in sets}, t_obs


def gene_set_test(
    model_setup,
    gene_sets,
    *,
    summary=("level", "slope"),
    method="rotation",
    n_rotations=9999,
    min_size=3,
    random_state=0,
):
    """Test whether the genes of a set change together between the conditions, with slides as replicates.

    Every slide contributes one level and one slope per gene (`replicate_summaries`), so the slide, not the
    cell, is the unit of replication. ``method="rotation"`` (default) is a self-contained rotation test (Wu
    et al. 2010, as limma's ``roast``): per gene a two-sample t, per set their (weighted) mean, compared
    with random rotations of the slides' residuals. Rotating all genes together keeps their correlation, so
    sets of co-regulated genes are not overconfident. ``method="ulm"`` runs decoupler's univariate linear
    model on the per-gene t values instead (a competitive test; needs ``pip install decoupler``).

    On random splits of the slides of three datasets the rotation test called 1 to 4% of sets at p < 0.05;
    sets of 10 genes with a 2% level shift were detected in 80% of cases, where single genes were not.

    Parameters
    ----------
    model_setup
        `setup_model`'s output for data binned with `sample`.
    gene_sets
        ``{name: [genes]}``, or a decoupler network (``source``, ``target`` and optionally ``weight``
        columns, e.g. ``decoupler.op.hallmark()``). Signed weights (e.g. PROGENy) make the set statistic a
        weighted mean, so genes expected to go down count with a negative sign.
    summary
        Which per-slide summaries to test: ``"level"``, ``"slope"`` or both.
    method
        ``"rotation"`` or ``"ulm"``.
    n_rotations
        Rotations for the rotation test; the smallest p-value is ``1 / (n_rotations + 1)``.
    min_size
        Sets with fewer genes in the data are skipped.
    random_state
        Seed for the rotations.

    Returns
    -------
    pandas.DataFrame
        One row per set and summary: ``size``, ``mean_t`` (positive: higher, or steeper, in condition 1),
        and for the rotation test ``pval_up``, ``pval_down``, ``pval_mixed`` (genes changing in either
        direction), ``pval`` (two-sided) and ``pval_adj`` (Benjamini-Hochberg over the sets of each summary).
        For ``method="ulm"``, ``mean_t`` is decoupler's score and ``pval`` its p-value as decoupler returns it
        (already adjusted across the sets).
    """
    level, slope, condition = replicate_summaries(model_setup)
    group = condition.to_numpy(dtype=float)
    if min(group.sum(), len(group) - group.sum()) < 2:
        raise ValueError("Gene-set tests need at least 2 slides per condition.")
    tables = {"level": level, "slope": slope}
    summaries = (summary,) if isinstance(summary, str) else tuple(summary)
    rng = np.random.default_rng(random_state)
    rows = []
    for name in summaries:
        M = tables[name]
        usable = M.columns[M.notna().all() & (M.std() > 0)]
        sets = {k: v for k, v in _as_sets(gene_sets, list(usable)).items() if len(v[0]) >= min_size}
        if not sets:
            warnings.warn(f"No gene set has {min_size} or more genes in the data.", UserWarning, stacklevel=2)
            continue
        Ymat = M[usable].to_numpy()
        if method == "rotation":
            res, _ = _rotation_pvalues(Ymat, group, sets, n_rotations, rng)
            for k, (mean_t, up, down, mixed) in res.items():
                rows.append({"summary": name, "set": k, "size": len(sets[k][0]), "mean_t": mean_t, "pval_up": up,
                             "pval_down": down, "pval_mixed": mixed, "pval": min(1.0, 2 * min(up, down))})  # fmt: skip
        elif method == "ulm":
            try:
                import decoupler as dc
            except ImportError as err:
                raise ImportError("method='ulm' needs decoupler: pip install decoupler") from err
            a, b = Ymat[group == 1], Ymat[group == 0]
            pooled = (((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)) / (len(group) - 2)
            t = (a.mean(0) - b.mean(0)) / np.sqrt(pooled * (1 / len(a) + 1 / len(b)))
            net = pd.DataFrame(
                [(k, usable[i], w) for k, (idx, wts) in sets.items() for i, w in zip(idx, wts, strict=True)],
                columns=["source", "target", "weight"],
            )
            es, pv = dc.mt.ulm(data=pd.DataFrame([t], columns=usable), net=net, tmin=min_size, verbose=False)
            for k in es.columns:
                rows.append({"summary": name, "set": k, "size": len(sets[k][0]), "mean_t": float(es.iloc[0][k]),
                             "pval": float(pv.iloc[0][k])})  # fmt: skip
        else:
            raise ValueError(f"Unknown method {method!r}; use 'rotation' or 'ulm'.")
    out = pd.DataFrame(rows)
    if len(out) and method == "rotation":
        out["pval_adj"] = out.groupby("summary")["pval"].transform(lambda p: _benjamini_hochberg(p.to_numpy()))
    return out.sort_values(["summary", "pval"]).reset_index(drop=True) if len(out) else out
