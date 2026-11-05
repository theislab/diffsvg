"""K-means clustering of the fitted curves, on their spline coefficients.

The features come from the tables `diffsvg.tl.fit_model` and
`diffsvg.tl.fit_profiles` write; the result is plotted by
`diffsvg.pl.plot_cluster_curves`.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _coefficient_table(adata, by, coefs_key, profiles_key):
    """The genes x coefficients table behind `by`, with every gene of `adata.var_names` (NaN where not fitted)."""

    def table(key, fit):
        if key not in adata.varm:
            raise KeyError(f"adata.varm[{key!r}] not found - run `diffsvg.tl.{fit}` first, with `inplace=True`.")
        return adata.varm[key]

    if by == "average":
        return table(f"{coefs_key}_beta", "fit_model") + table(f"{coefs_key}_beta_1", "fit_model") / 2.0
    if by == "interaction":
        return table(f"{coefs_key}_beta_1", "fit_model")
    if by == "profile":
        return table(f"{profiles_key}_beta", "fit_profiles")
    raise ValueError(f"Unknown by {by!r}. Available: ['average', 'interaction', 'profile']")


def _features(table, scale):
    """Coefficient rows as k-means features: mean coefficient removed, optionally scaled to unit length.

    The columns of the (centred, intercept-including) B-spline basis sum to 0,
    so adding a constant to a gene's coefficients does not change its curve:
    that direction holds only the prior and the sampling noise, not data.
    """
    values = table.to_numpy(dtype=float)
    values = values - values.mean(axis=1, keepdims=True)
    if scale:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.where(norms > 0, norms, 1.0)
    return values


def cluster_curves(
    adata,
    by="average",
    n_clusters=5,
    genes=None,
    scale=False,
    coefs_key="dsvg_coefs",
    profiles_key="dsvg_profiles",
    key_added=None,
    random_state=0,
    n_init=100,
    inplace=True,
):
    """Group genes by the shape of their fitted curves, with k-means on the spline coefficients.

    Parameters
    ----------
    adata
        Annotated data matrix that `diffsvg.tl.fit_model` (or
        `diffsvg.tl.fit_profiles`, for ``by="profile"``) wrote into.
    by
        Which curves to compare:

        - ``"average"``: ``beta + beta_1 / 2``, the combined curve: the mean
          of the two condition curves, weighting the conditions equally. Not
          `beta` alone, which with the 0/1 condition coding is the curve of
          condition 0.
        - ``"interaction"``: the interaction term `beta_1`, i.e. the shape of
          the difference curve ``f1 - f0``. Not the level difference, which
          is a separate coefficient (``categorical_beta``).
        - ``"profile"``: `beta` of `diffsvg.tl.fit_profiles`, the one curve
          of a fit that ignores the condition. Without slide effects.
    n_clusters
        Number of clusters (`k`).
    genes
        Genes to cluster, by default every fitted one.
    scale
        Scale each gene's coefficients to unit length, so that genes group by
        shape alone, however strong the trend. By default (`False`) the
        amplitude counts, so genes without a trend share one flat cluster.
        Scaling blows the noise of such genes up to the size of real trends,
        so combine it with `genes`, e.g. the genes with a clear trend.
    coefs_key, profiles_key
        Prefixes passed to `diffsvg.tl.fit_model` / `diffsvg.tl.fit_profiles`.
    key_added
        Where to write the result. Defaults to ``f"dsvg_cluster_{by}"``.
    random_state, n_init
        Passed to `sklearn.cluster.KMeans`. Many restarts cost next to nothing
        with one row per gene, and they make the result depend far less on the
        seed: on an 88-gene panel 10 gave up to 25 different partitions across
        30 seeds, 100 mostly one.
    inplace
        If `True` (default), write ``adata.var[key_added]`` (categorical,
        `NaN` for genes not clustered) and ``adata.uns[key_added]`` (`by`,
        `scale`, `n_clusters`, `silhouette`, `inertia`, `centers`) and return
        `None`. Otherwise return the labels as a `pandas.Series`.

    Returns
    -------
    None or pandas.Series
        Labels are ``"0"``, ``"1"``, ..., the largest cluster first.

    Examples
    --------
    >>> dvg.tl.cluster_curves(adata, by="average", n_clusters=6)
    >>> dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_average")
    """
    table = _coefficient_table(adata, by, coefs_key, profiles_key)
    if genes is not None:
        genes = list(genes)
        missing = [g for g in genes if g not in table.index]
        if missing:
            raise KeyError(f"genes not found in adata.var_names: {missing}")
        table = table.loc[genes]
    table = table.dropna(how="any")
    if len(table) < n_clusters:
        raise ValueError(f"Cannot form {n_clusters} clusters from {len(table)} fitted genes.")

    features = _features(table, scale)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit(features)

    # Largest cluster first; labels do not depend on KMeans's arbitrary numbering.
    sizes = np.bincount(kmeans.labels_, minlength=n_clusters)
    order = np.argsort(-sizes, kind="stable")
    new_label = np.empty(n_clusters, dtype=int)
    new_label[order] = np.arange(n_clusters)
    categories = [str(c) for c in range(n_clusters)]
    labels = pd.Series(
        pd.Categorical([str(c) for c in new_label[kmeans.labels_]], categories=categories),
        index=table.index,
        name=key_added or f"dsvg_cluster_{by}",
    )

    if not inplace:
        return labels

    # A single label, or one gene per cluster, has no silhouette.
    has_silhouette = 1 < labels.nunique() < len(labels)
    key = labels.name
    adata.var[key] = labels.reindex(adata.var_names)
    adata.uns[key] = {
        "by": by,
        "scale": bool(scale),
        "n_clusters": int(n_clusters),
        "silhouette": float(silhouette_score(features, kmeans.labels_)) if has_silhouette else float("nan"),
        "inertia": float(kmeans.inertia_),
        "centers": kmeans.cluster_centers_[order],
    }
    return None
